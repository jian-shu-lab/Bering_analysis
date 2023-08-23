import os
import sys
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile as tiff

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 16

sys.path.append('/data/aronow/Kang/spatial/Bering/benchmark/bm_code/')
import src as bm

markers_nontumor = ['CCR7', 'CD14', 'CD53', 'CD68', 'CD74', 'CD163', 'COL3A1', 'COL4A1', 'COL4A2', 'COL5A1', 'COL6A3', 'SOD2', 'SRGN', 'TAGLN', 'VIM']
markers_tumor = ['CEACAM6', 'DDR1', 'EPCAM', 'EPHA2', 'ITGB4', 'KRT8', 'KRT17', 'KRT19', 'S100P', 'SLC2A1', 'TM4SF1']

def plot_gene_pairs(adata, gene1, gene2, save_name = None):
    # adata is raw counts
    x = adata[:, gene1].X.toarray().flatten()
    y = adata[:, gene2].X.toarray().flatten()
    fig, ax = plt.subplots(figsize = (4, 4))
    ax.scatter(x, y, s = 0.5, color = '#3399FF')
    ax.set_xlabel(gene1)
    ax.set_ylabel(gene2)
    xmax = np.percentile(x, 95)
    ymax = np.percentile(y, 95)
    ax.set_xlim([-0.08, xmax])
    ax.set_ylim([-0.08, ymax])
    ax.grid(False)
    
    if save_name is not None:
        fig.savefig(save_name, dpi = 300, bbox_inches = 'tight')

def draw_heatmap(adata, gene_list_1, gene_list_2, savename = None):
    # normalized values are recommended
    df = adata.obs.copy()
    gene_list = gene_list_1 + gene_list_2
    df[gene_list] = adata[:, gene_list].X.toarray()
    df_mean = df.groupby('labels').mean()[gene_list]
    
    # sort clusters by gene_list_1
    df_mean_v2 = df_mean.copy()
    df_mean_v2['gene_list_1_avg'] = df_mean_v2[gene_list_1].mean(axis = 1)
    df_mean_v2 = df_mean_v2.sort_values(by = 'gene_list_1_avg', ascending = False)
    df_mean = df_mean.loc[df_mean_v2.index, :]
    
    # sort genes by nontumor/tumor clusters
    nontumor_clusters = df_mean_v2[df_mean_v2['gene_list_1_avg'] > 0.50].index.values
    tumor_clusters = df_mean_v2[df_mean_v2['gene_list_1_avg'] < 0.50].index.values

    # non tumor genes and clusters
    df_mean_v3 = df_mean.loc[nontumor_clusters, :].copy().T
    df_mean_v3 = df_mean_v3.loc[gene_list_1, :]
    df_mean_v3['nontumor_avg'] = df_mean_v3.mean(axis = 1)
    gene_list_1_reorder = df_mean_v3.sort_values(by = 'nontumor_avg', ascending = False).index.values

    df_mean_v3 = df_mean.loc[nontumor_clusters, :].copy()
    df_mean_v3 = df_mean_v3.loc[:, gene_list_1]
    df_mean_v3['genelist_1_avg'] = df_mean_v3.mean(axis = 1)
    nontumor_reorder = df_mean_v3.sort_values(by = 'genelist_1_avg', ascending = False).index.values
    
    # tumor genes and clusters
    df_mean_v3 = df_mean.loc[tumor_clusters, :].copy().T
    df_mean_v3 = df_mean_v3.loc[gene_list_2, :]
    df_mean_v3['tumor_avg'] = df_mean_v3.mean(axis = 1)
    gene_list_2_reorder = df_mean_v3.sort_values(by = 'tumor_avg', ascending = True).index.values

    df_mean_v3 = df_mean.loc[tumor_clusters, :].copy()
    df_mean_v3 = df_mean_v3.loc[:, gene_list_2]
    df_mean_v3['genelist_2_avg'] = df_mean_v3.mean(axis = 1)
    tumor_reorder = df_mean_v3.sort_values(by = 'genelist_2_avg', ascending = False).index.values

    # combine reordered genes and clusters
    gene_list_reorder = list(gene_list_1_reorder) + list(gene_list_2_reorder)
    cluster_reorder = list(nontumor_reorder) + list(tumor_reorder)
    df_mean = df_mean.loc[:, gene_list_reorder]
    df_mean = df_mean.loc[cluster_reorder, :]

    fig, ax = plt.subplots(figsize = (12, 4))
    sns.heatmap(df_mean, cmap = 'RdBu_r', vmin = 0, vmax = 5, ax = ax)
    # sns.clustermap(df_mean, cmap = 'RdBu_r', vmin = 0, vmax = 10)
    ax.set_xticks(np.arange(len(df_mean.columns)) + 0.5)
    ax.set_yticks(np.arange(len(df_mean.index.values)) + 0.5)
    ax.set_xticklabels(df_mean.columns, rotation = 45, ha = 'right', va = 'top', fontsize = 16)
    ax.set_yticklabels(df_mean.index.values, rotation = 0)
    if savename is not None:
        fig.savefig(savename, dpi = 300, bbox_inches = 'tight')

def draw_correlation(adata, gene_list, savename = None):
    from scipy.stats import spearmanr
    adata = adata[:, gene_list].copy()
    print(adata.shape)
    clusters = np.unique(adata.obs['labels'])
    corr_mtx = np.zeros((len(clusters), len(clusters)))
    for idx1, cluster1 in enumerate(clusters):
        for idx2, cluster2 in enumerate(clusters):
            x = adata[adata.obs['labels'] == str(cluster1), :].X.toarray().mean(axis = 0)
            y = adata[adata.obs['labels'] == str(cluster2), :].X.toarray().mean(axis = 0)
            corr = spearmanr(x, y)[0]
            corr_mtx[idx1, idx2] = corr
    sns.set(font_scale = 1.0)
    
    # 0. random dendrogram (deprecated)
    # df_corr = pd.DataFrame(corr_mtx, index = clusters, columns = clusters)
    # sns.clustermap(df_corr, cmap = 'RdBu_r', vmin = 0, vmax = 1, figsize = (8, 8))

    # 1. order from "Raw"'s dendrogram
    orders = ['Tumor', 'Epithelial', 'Endothelial', 'Fibroblast', 'CD8+ T', 'Neutrophil', 'NK', 'B', 'CD4+ T', 'pDC', 'Macrophage', 'mDC', 'Mast', 'Plasmablast']
    orders = [i for i in orders if i in clusters]
    df_corr = pd.DataFrame(corr_mtx, index = clusters, columns = clusters)
    df_corr = df_corr.loc[orders, orders]

    fig, ax = plt.subplots(figsize = (4, 4))
    sns.heatmap(df_corr, cmap = 'RdBu_r', vmin = 0, vmax = 1, square = True, cbar_kws = {'shrink': 0.5}, ax = ax)
    ax.set_xticks(np.arange(len(df_corr.columns)) + 0.5)
    ax.set_yticks(np.arange(len(df_corr.index.values)) + 0.5)
    ax.set_xticklabels(df_corr.columns, rotation = 90, ha = 'right', va = 'top', fontsize = 12)
    ax.set_yticklabels(df_corr.index.values, rotation = 0, fontsize = 12)
    
    if savename is not None:
        fig.savefig(savename, dpi = 300, bbox_inches = 'tight')

if __name__ == '__main__':
    if not os.path.exists('figures'):
        os.mkdir('figures')
    if not os.path.exists('figures/tumor_marker_heatmap'):
        os.mkdir('figures/tumor_marker_heatmap')
    if not os.path.exists('figures/correlation'):
        os.mkdir('figures/correlation')

    # 1. survey in single cell data
    combined = sc.read('../adata_combined_raw.h5ad')
    sources = ['Raw', 'Watershed', 'Cellpose', 'ClusterMap (no image)', 'ClusterMap (w. image)', 'Baysor (no prior)', 'Baysor (w. prior)', 'Bering']

    # ## 2. get marker genes (heatmap)
    # for source in sources:
    #     print(source)

    #     raw = combined[combined.obs.source == source, :].copy()
    #     sc.pp.filter_cells(raw, min_counts = 10)
    #     sc.pp.normalize_total(raw, target_sum = 1e3)

    #     processed = sc.read(f'adata_processed_{source}_transfer_simplified.h5ad')
    #     raw = raw[processed.obs.index.values, :].copy()
    #     processed.X = raw.X.copy()
        
    #     savename = f'figures/tumor_marker_heatmap/heatmap_{source}.pdf'
    #     draw_heatmap(processed, gene_list_1=markers_nontumor, gene_list_2=markers_tumor, savename=savename)

    ### 3. draw cluster correlations
    df_deg = pd.read_csv('/data/aronow/Kang/spatial/Bering/benchmark/run/bm2_cosmx_lung_tumor/output/DEGs_Raw.txt', sep = '\t', header = 0, index_col = 0)
    df_deg = df_deg[(df_deg['pvals_adj'] < 0.05) & (df_deg['logfoldchanges'] > 1)].copy()
    total_genes = np.unique(df_deg.index.values)
    for source in sources:

        raw = combined[combined.obs.source == source, :].copy()
        sc.pp.filter_cells(raw, min_counts = 10)
        sc.pp.normalize_total(raw, target_sum = 1e3)
        
        processed = sc.read(f'adata_processed_{source}_transfer_simplified.h5ad')
        raw = raw[processed.obs.index.values, :].copy()
        processed.X = raw.X.copy()

        # savename = f'figures/correlation/correlation_{source}.pdf' # random dendrogram
        savename = f'figures/correlation/correlation_v2_{source}.pdf' # order from "Raw"'s dendrogram
        draw_correlation(processed, gene_list = total_genes, savename = savename)