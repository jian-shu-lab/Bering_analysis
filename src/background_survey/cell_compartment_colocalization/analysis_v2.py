import os
import numpy as np 
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

n_spots_list = [5, 10, 15, 20, 25, 30]
for n_spots in n_spots_list:
    adata = sc.read(f'results/NCV_all_cells_with_noise_n_neighbors_{n_spots}_processed.h5ad')

    fig, axes = plt.subplots(ncols = 2, nrows = 1, figsize=(18, 7))
    sc.pl.umap(adata, color = 'labels', size = 20, legend_fontsize = 14, ax = axes[0])
    sc.pl.umap(adata, color = 'compartments', size = 20, legend_fontsize = 14, ax = axes[1])
    
    # set font sizes for axis labels and title
    axes[0].set_xlabel('UMAP1', fontsize = 35)
    axes[0].set_ylabel('UMAP2', fontsize = 35)
    axes[0].set_title('')
    # plot legend on right
    axes[0].legend(loc = 'right', bbox_to_anchor = (1.45, 0.5), fontsize = 14, markerscale = 1.5)

    axes[1].set_xlabel('UMAP1', fontsize = 35)
    axes[1].set_ylabel('UMAP2', fontsize = 35)
    axes[1].set_title('')
    axes[1].legend(loc = 'lower right', bbox_to_anchor = (1.00, 0.0), fontsize = 18, markerscale = 2.5)

    # set gap
    plt.subplots_adjust(wspace = 0.55)
    
    fig.savefig(f'results/umap_labels_compartments_NCV_{n_spots}_v2.png', dpi = 300, bbox_inches='tight')