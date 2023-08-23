import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import tifffile as tiff
from shapely.geometry import Polygon

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

    dots = np.stack([x, y], axis=1)
    hull = ConvexHull(dots)
    polygon = Polygon(dots[hull.vertices, :])

    # fig, ax = plt.subplots(1, len(selected_features)+2, figsize=((len(selected_features)+2) * 3, 3), sharex=True, sharey=True)
    # get the largest cell in the mask
    dapi_cell = dapi[int(ymin):int(ymax), int(xmin):int(xmax)]
    dapi_cell = dapi_cell[None, :, :]
    mask = run_cellpose(dapi_cell, diameter = None)
    nonzero_values, counts = np.unique(mask[mask != 0], return_counts=True)
    max_count_index = np.argmax(counts)
    dominant_cell = nonzero_values[max_count_index]
    mask[mask!=dominant_cell] = 0

    nonzero_nuclei = np.argwhere(mask)
    hull_nuclei = ConvexHull(nonzero_nuclei)
    polygon_nuclei = Polygon(nonzero_nuclei[hull_nuclei.vertices, :])
    overlap_polygon = polygon.intersection(polygon_nuclei)

    fig, ax = plt.subplots(1, len(selected_features), figsize=((len(selected_features)) * 3, 3), sharex=True, sharey=True)
    for i in range(len(selected_features)):
        xf, yf = x[features == selected_features[i]], y[features == selected_features[i]]
        ax[i].scatter(xf, yf, s=1, c='#CC6600')

        xs, ys = polygon.exterior.xy
        # ax[i].plot(dots[hull.vertices, 0], dots[hull.vertices, 1], 'k-', lw = 0.4)
        # ax[i].fill(dots[hull.vertices, 0], dots[hull.vertices, 1], '#F3F189', alpha = 0.5)

        ax[i].plot(xs, ys, 'k--', lw = 0.4)
        ax[i].fill(xs, ys, '#F3F189', alpha=0.5)

        # ax[i].plot(nonzero_nuclei[hull_nuclei.vertices, 0], nonzero_nuclei[hull_nuclei.vertices, 1], 'k--', lw = 0.4)
        # ax[i].fill(nonzero_nuclei[hull_nuclei.vertices, 0], nonzero_nuclei[hull_nuclei.vertices, 1], '#CBA098', alpha = 0.5)
        xs, ys = overlap_polygon.exterior.xy
        ax[i].plot(xs, ys, 'k--', lw = 0.4)
        ax[i].fill(xs, ys, '#CBA098', alpha=0.5)

        ax[i].set_title(f'{selected_features[i]}')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xlim(0, xmax-xmin)
        ax[i].set_ylim(0, ymax-ymin)

    # dapi_cell = dapi[int(ymin):int(ymax), int(xmin):int(xmax)]
    # ax[-2].imshow(dapi_cell, cmap='gray')

    # dapi_cell = dapi_cell[None, :, :]
    # mask = run_cellpose(dapi_cell, diameter = None)
    # print(mask.shape)
    # with open('mask.pkl', 'wb') as f:
    #     pickle.dump(mask, f)
    # ax[-1].imshow(mask, cmap='gray')

    # return fig
    fig.savefig(f'figures/subcellular_distribution/cell_{cell_id}_combined_v2.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    np.random.seed(0)
    cell_types = ['tumor 12', 'tumor 5']
    n_cells = 72
    df_spots = pd.read_csv('/data/aronow/Kang/spatial/Bering/benchmark/bm_data/bm2_lungtumor_cosmx_he_et_al/Lung5_Rep1_tx_fov_10.txt',sep='\t',header=0,index_col=0)
    dapi = tiff.imread('/data/aronow/Kang/spatial/Bering/benchmark/bm_data/bm2_lungtumor_cosmx_he_et_al/F010/Lung5_Rep1_DAPI.tif')
    df_spots = df_spots[df_spots['labels'].isin(cell_types)].copy()
    # random_cells = np.random.choice(df_spots.segmented.unique(), size = n_cells, replace = False)
    # random_cells = np.setdiff1d(random_cells, [0, -1])
    random_cells = [21, 35, 64, 72, 99, 104, 121, 134]
    df_spots = df_spots[df_spots['segmented'].isin(random_cells)].copy()
    
    nuclei_genes = ['MALAT1', 'NEAT1', 'MZT2A']
    membrane_genes = ['OLFM4','DUSP5','S100A6']
    
    '''
    for cell_id in [17, 21, 26, 35, 64, 72, 134]:
        # # nuclei genes
        # fig = _visualize_genes(df_spots, cell_id, nuclei_genes)
        # fig.savefig(f'figures/subcellular_distribution/cell_{cell_id}_nuclei.png', dpi=300, bbox_inches='tight')

        # # membrane genes
        # fig = _visualize_genes(df_spots, cell_id, membrane_genes)
        # fig.savefig(f'figures/subcellular_distribution/cell_{cell_id}_membrane.png', dpi=300, bbox_inches='tight')

        # combined
        fig = _visualize_genes(df_spots, cell_id, nuclei_genes + membrane_genes, dapi)
    '''

    # multiprocessing
    # cell_ids = [17, 21, 26, 35, 64, 72, 134]
    cell_ids = random_cells
    # cell_ids = [21, 35, 64, 72, 99, 104, 121, 134]
    pool = mp.Pool(8)
    for cell_id in cell_ids:
        pool.apply_async(_visualize_genes, args=(df_spots, cell_id, nuclei_genes, membrane_genes, dapi))
    pool.close()
    pool.join()