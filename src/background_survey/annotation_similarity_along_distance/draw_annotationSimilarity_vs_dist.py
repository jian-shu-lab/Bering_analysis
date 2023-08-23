import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 18
mpl.rcParams['font.family'] = 'Arial'

def load_data():
    df_spots_seg = pd.read_csv('/data/aronow/Kang/spatial/Bering/validation/run_bm1/slice_21/mouse1_slice21_spots_v2.tsv', sep = '\t', header = 0, index_col = 0)
    df_spots_unseg = pd.read_csv('/data/aronow/Kang/spatial/Bering/validation/run_bm1/slice_21/mouse1_slice21_spots_unsegmented_v2.tsv', sep = '\t', header = 0, index_col = 0)
    df_spots_unseg['segmented'] = -1
    df_spots_unseg['labels'] = 'background'
    df_spots_all = pd.concat([df_spots_seg, df_spots_unseg], axis = 0)
    return df_spots_seg, df_spots_unseg, df_spots_all

def get_probability_same_cells_types(df_spots, num_edges = 500000):
    x, y = df_spots.x.values, df_spots.y.values
    labels = df_spots.labels.values
    cells = df_spots.segmented.values

    node_indices = np.arange(len(df_spots))
    # get random edges 
    edge_indices = np.random.choice(node_indices, size = (num_edges, 2), replace = True)
    # remove self-loops
    edge_indices = edge_indices[edge_indices[:, 0] != edge_indices[:, 1]]
    # get distances between edges
    dists = np.sqrt((x[edge_indices[:, 0]] - x[edge_indices[:, 1]])**2 + (y[edge_indices[:, 0]] - y[edge_indices[:, 1]])**2)
    # get labels
    labels = df_spots.labels.values
    # get same-type edges
    same_type = labels[edge_indices[:, 0]] == labels[edge_indices[:, 1]]
    # get same-cell edges
    same_cell = cells[edge_indices[:, 0]] == cells[edge_indices[:, 1]]
    return same_type, same_cell, dists
    
def draw_scatterplot(same_type, same_cell, dists, max_dist = 200):
    # remove edges with distance > max_dist
    indices = np.where(dists <= max_dist)[0]
    same_type = same_type[indices]
    same_cell = same_cell[indices]
    dists = dists[indices]

    # 
    prob_same_type = []
    prob_same_cell = []
    dist_list = []
    for dist_thresh in np.linspace(0, max_dist, 40):
        dist_list += [dist_thresh]
        subindices = np.where((dists <= dist_thresh + 0.5) & (dists > dist_thresh - 0.5))[0]
        prob_same_type.append(np.sum(same_type[subindices]) / len(subindices))
        prob_same_cell.append(np.sum(same_cell[subindices]) / len(subindices))

    # draw probability of same cell type vs distance
    fig, ax = plt.subplots(figsize = (3, 3))
    # show dots on the line
    ax.plot(dist_list, prob_same_type, color = 'red', label = 'same cell type', marker = 'o', linestyle = '-')
    ax.plot(dist_list, prob_same_cell, color = 'blue', label = 'same cell', marker = 'o', linestyle = '-')
    ax.set_xlabel('Distance (um)')
    ax.set_ylabel('Probability')
    ax.legend()
    fig.savefig('annotationSimilarity_vs_dist.pdf', dpi = 300)


if __name__ == '__main__':
    df_spots_seg, df_spots_unseg, df_spots_all = load_data()
    same_type, same_cell, dists = get_probability_same_cells_types(df_spots_seg)
    draw_scatterplot(same_type, same_cell, dists)