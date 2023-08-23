import pickle
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import tifffile as tiff
from shapely.geometry import Polygon, Point

import logging
import multiprocessing as mp
from cellpose import models, core, plot
from cellpose.io import logger_setup
logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1

def run_cellpose(img, channel = [0,0], model = 'nuclei', diameter = None, flow_threshold = None, plot_name = None):
    # mpl.rcParams['figure.dpi'] = 300
    use_GPU = core.use_gpu()
    
    # DEFINE CELLPOSE MODEL
    model = models.Cellpose(gpu=use_GPU, model_type=model)
    masks, flows, styles, diams = model.eval(img, diameter=diameter, flow_threshold=flow_threshold, channels=channel)

    nimg = 1
    return masks

def _visualize_genes(df_spots, cell_id, nuclei_features, cytoplasm_features, dapi):
    df_cell = df_spots[df_spots['segmented'] == cell_id].copy()
    x, y = df_cell['x'].values, df_cell['y'].values
    features = df_cell['features'].values

    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    x = x - xmin
    y = y - ymin

    # polygon of whole cell
    dots = np.stack([x, y], axis=1)
    hull = ConvexHull(dots)
    polygon = Polygon(dots[hull.vertices, :])

    # polygon of nuclei
    dapi_cell = dapi[int(ymin):int(ymax), int(xmin):int(xmax)]
    dapi_cell = dapi_cell[None, :, :]
    mask = run_cellpose(dapi_cell, diameter = None)
    nonzero_values, counts = np.unique(mask[mask != 0], return_counts=True)
    max_count_index = np.argmax(counts)
    dominant_cell = nonzero_values[max_count_index]
    mask[mask!=dominant_cell] = 0
    mask = mask.T

    # overlap polygon
    nonzero_nuclei = np.argwhere(mask)
    hull_nuclei = ConvexHull(nonzero_nuclei)
    polygon_nuclei = Polygon(nonzero_nuclei[hull_nuclei.vertices, :])
    overlap_polygon = polygon.intersection(polygon_nuclei)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
    gene_to_col = {'MALAT1': '#F4A460', 'NEAT1': '#D2B48C', 'MZT2A': '#A0522D', 'OLFM4': '#5F9EA0', 'DUSP5': '#4682B4', 'S100A6': '#00BFFF'}
    selected_features = list(nuclei_features) + list(cytoplasm_features)
    for selected_feature in selected_features:
        xf, yf = x[features == selected_feature], y[features == selected_feature]
        ax.scatter(xf, yf, s=3.5, c=gene_to_col[selected_feature], label=selected_feature)

    xs, ys = polygon.exterior.xy
    ax.plot(xs, ys, 'k-', lw = 0.4)
    ax.fill(xs, ys, '#F3F189', alpha=0.5)

    xs, ys = overlap_polygon.exterior.xy
    ax.plot(xs, ys, 'k-', lw = 0.4)
    ax.fill(xs, ys, '#CBA098', alpha=0.3)

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.legend(markerscale=5, fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(0, xmax-xmin)
    ax.set_ylim(0, ymax-ymin)
    plt.axis('off')

    # fig.savefig(f'figures/subcellular_distribution/cell_{cell_id}_combined_v2.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'figures/subcellular_distribution/cell_{cell_id}_combined_v3.png', dpi=300, bbox_inches='tight')

    # run statistics
    # get furthest points
    vertices = list(polygon.exterior.coords)
    distances = [Point(point1).distance(Point(point2)) for point1, point2 in combinations(vertices, 2)]
    largest_distance = sorted(distances, reverse=True)[0]
    radius = largest_distance / 2

    df_tps_dist = pd.DataFrame(columns=['gene', 'cell', 'distance'])
    if mask.max() != 0:
        nuclei_cx, nuclei_cy = np.mean(nonzero_nuclei[:, 0]), np.mean(nonzero_nuclei[:, 1])
        for selected_feature in (nuclei_features+cytoplasm_features):
            xf, yf = x[features == selected_feature], y[features == selected_feature]
            for i in range(len(xf)):
                distance = np.sqrt((xf[i] - nuclei_cx)**2 + (yf[i] - nuclei_cy)**2)
                df_tps_dist = df_tps_dist.append({'gene': selected_feature, 'cell': cell_id, 'distance': distance}, ignore_index=True)
    return df_tps_dist

if __name__ == '__main__':

    np.random.seed(0)
    cell_types = ['tumor 12', 'tumor 5']
    n_cells = 300
    df_spots = pd.read_csv('/data/aronow/Kang/spatial/Bering/benchmark/bm_data/bm2_lungtumor_cosmx_he_et_al/Lung5_Rep1_tx_fov_10.txt',sep='\t',header=0,index_col=0)
    dapi = tiff.imread('/data/aronow/Kang/spatial/Bering/benchmark/bm_data/bm2_lungtumor_cosmx_he_et_al/F010/Lung5_Rep1_DAPI.tif')
    df_spots = df_spots[df_spots['labels'].isin(cell_types)].copy()
    random_cells = np.random.choice(df_spots.segmented.unique(), size = n_cells, replace = False)
    random_cells = np.setdiff1d(random_cells, [0, -1])
    df_spots = df_spots[df_spots['segmented'].isin(random_cells)].copy()
    
    nuclei_genes = ['MALAT1', 'NEAT1', 'MZT2A']
    membrane_genes = ['OLFM4','DUSP5','S100A6']
    
    # multiprocessing
    df_stats_all = pd.DataFrame()
    cell_ids = random_cells
    for cell_id in cell_ids:
        try:
            df_stats = _visualize_genes(df_spots, cell_id, nuclei_genes, membrane_genes, dapi)
            print(f'cell {cell_id} generate table with {len(df_stats)} rows')
            df_stats_all = pd.concat([df_stats_all, df_stats], axis = 0)
        except:
            continue

    df_stats_all.to_csv('distance.csv', sep = ',', header = True, index = False)