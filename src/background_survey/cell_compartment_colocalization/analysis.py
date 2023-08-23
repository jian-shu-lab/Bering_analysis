import os
import numpy as np 
import pandas as pd
import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

n_spots_list = [5, 10, 15, 20, 25, 30]
for n_spots in n_spots_list:
    adata = sc.read(f'compartment_data/NCV_all_cell_compartments_n_cells_2048_n_neighbors_{n_spots}.h5ad')
    
    adata.obs['compartments'] = [i.split('_')[1] for i in adata.obs.index]
    adata.obs['labels'] = [i.split('_')[2] for i in adata.obs.index.values]
    
    # sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)

    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors = 20, n_pcs = 10)
    sc.tl.umap(adata)

    sc.tl.leiden(adata)
    sc.pl.umap(adata, color = 'leiden', save = f'_leiden_{n_spots}.png')
    sc.pl.umap(adata, color = 'labels', save = f'_labels_{n_spots}.png')
    sc.pl.umap(adata, color = 'compartments', save = f'_compartments_{n_spots}.png')
    adata.write(f'results/NCV_all_cells_with_noise_n_neighbors_{n_spots}_processed.h5ad')

    # dpi = 300
    # sc.pl.umap(adata, color = 'leiden', legend_loc = 'on data', save = f'_leiden_ondata_NCV_noise_n_spots_{n_spots}.png', dpi = 300)
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    sc.pl.umap(adata, color = 'labels', legend_loc = 'on data', size = 20, legend_fontsize = 14, ax = ax)
    # set font sizes for axis labels and title
    ax.set_xlabel('UMAP1', fontsize = 35)
    ax.set_ylabel('UMAP2', fontsize = 35)
    # no title
    ax.set_title('')
    
    fig.savefig(f'results/umap_labels_ondata_NCV_noise_n_spots_{n_spots}.png', dpi = 300)