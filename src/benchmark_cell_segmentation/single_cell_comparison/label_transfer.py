import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. plot 
labels = ['Raw', 'Watershed', 'Cellpose', 'ClusterMap (no image)', 'ClusterMap (w. image)', 'Baysor (no prior)', 'Baysor (w. prior)', 'Bering']
h5ad_files = [f'../adata_processed_{label}.h5ad' for label in labels]

# tumor 5         833
# neutrophil      462
# fibroblast      335
# macrophage      330
# endothelial     318
# epithelial      171
# mast            125
# plasmablast     120
# T CD4 naive     108
# mDC              81
# B-cell           75
# NK               64
# pDC              48
# monocyte         23
# tumor 12         19
# T CD8 naive      14
# Treg              2
# T CD4 memory      1

cell_type_dict = {
    'tumor 5': 'Tumor',
    'neutrophil': 'Neutrophil',
    'fibroblast': 'Fibroblast',
    'macrophage': 'Macrophage',
    'endothelial': 'Endothelial',
    'epithelial': 'Epithelial',
    'mast': 'Mast',
    'plasmablast': 'Plasmablast',
    'T CD4 naive': 'CD4+ T',
    'mDC': 'mDC',
    'B-cell': 'B',
    'NK': 'NK',
    'pDC': 'pDC',
    'monocyte': 'Macrophage',
    'tumor 12': 'Tumor',
    'T CD8 naive': 'CD8+ T',
    'Treg': 'CD4+ T',
    'T CD4 memory': 'CD4+ T'
}

type_to_col = {
    'Tumor': '#DBDB8D',
    'Epithelial': '#B4BC61',
    'Fibroblast': '#17BECF',
    'Endothelial': '#E377C2',
    'Neutrophil': '#C5B0D5',
    'Macrophage': '#FFBB78',
    'mDC': '#AEC7E8',
    'Mast': '#97DF8A',
    'CD4+ T': '#D51F21',
    'CD8+ T': '#A940FC',
    'NK': '#FF7F0D',
    'B': '#1F77B4',
    'pDC': '#C49C94',
    'Plasmablast': '#F7B6D2',
}

def draw_umap(adata, figname, color_by = 'labels'):
    cell_types = adata.obs[color_by].values
    umap1, umap2 = adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1]
    fig, ax = plt.subplots(figsize = (3.5, 3.5))
    for ct, col in type_to_col.items():
        umap1_sub = umap1[cell_types == ct]
        umap2_sub = umap2[cell_types == ct]
        ax.scatter(umap1_sub, umap2_sub, c = col, label = ct, s = 1.0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale = 5)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    fig.savefig(figname, dpi = 300, bbox_inches='tight')

for label, h5ad_file in zip(labels, h5ad_files):
    adata = sc.read(h5ad_file)
    sc.pl.umap(adata, color='leiden', save=f'_leiden_{label}.png')
    if label == 'Raw':
        # sc.pl.umap(adata, color = 'labels', save=f'_labels_{label}.png')
        
        adata.obs['labels'] = [cell_type_dict[x] for x in adata.obs['labels']]
        # sc.pl.umap(adata, color = 'labels', save=f'_labels_{label}_simplified.png')
        draw_umap(adata, figname = f'figures/umap_labels_{label}_simplified_v2.png', color_by = 'labels')
        adata.write(f'adata_processed_{label}_transfer_simplified.h5ad')

# 2. label transfer
adata_ref = sc.read('../adata_processed_Raw.h5ad')
adata_ref.obs['labels'] = [cell_type_dict[x] for x in adata_ref.obs['labels']] # v2

labels_query = ['Watershed', 'Cellpose', 'ClusterMap (no image)', 'ClusterMap (w. image)', 'Baysor (no prior)', 'Baysor (w. prior)', 'Bering']
h5ad_files_query = [f'../adata_processed_{label}.h5ad' for label in labels_query]
for label, h5ad_file in zip(labels_query, h5ad_files_query):
    adata_query = sc.read(h5ad_file)
    adata_query.obsm['X_umap_ori'] = adata_query.obsm['X_umap'].copy()
    sc.tl.ingest(adata_query, adata_ref, obs='labels')
    adata_query.obsm['X_umap'] = adata_query.obsm['X_umap_ori'].copy()
    
    # sc.pl.umap(adata_query, color='labels', save=f'_labels_transfer_{label}.png')
    # adata_query.write(f'adata_processed_{label}_transfer.h5ad')

    # sc.pl.umap(adata_query, color='labels', save=f'_labels_transfer_{label}_simplified.png')
    draw_umap(adata_query, figname = f'figures/umap_labels_{label}_simplified_v2.png', color_by = 'labels')
    adata_query.write(f'adata_processed_{label}_transfer_simplified.h5ad')
    