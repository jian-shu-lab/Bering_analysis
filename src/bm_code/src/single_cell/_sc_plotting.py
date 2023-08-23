import numpy as np
import pandas as pd 
import scanpy as sc

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

from ._sc_degs import get_DEGs

def draw_stacked_violin_plot():
    1

def draw_heatmap_DEGs(
    adata, groupby = 'labels', groups = 'all', reference = 'rest', method = 'wilcoxon',
    padj_thresh = 0.05, n_top_genes = None, 
    save = True, output_name = None
):
    '''
    Draw heatmap of DEGs
    '''
    
    df_deg = get_DEGs(adata, groupby = groupby, groups = groups, reference = reference, method = method)
    df_deg = df_deg[(df_deg['pvals_adj'] < padj_thresh) & (df_deg['logfoldchanges'] > 0)]

    if n_top_genes is not None:
        # draw top n genes per cluster
        data_all = pd.DataFrame()
        for i in adata.obs[groupby].cat.categories:
            df = df_deg[df_deg['Status'] == i].sort_values(by='pvals_adj')
            df = df.iloc[:n_top_genes, :]
            data = df.loc[:, adata.obs[groupby].cat.categories]
            data_all = pd.concat([data_all, data], axis=0)
    else:
        data_all = df_deg.loc[:, adata.obs[groupby].cat.categories]

    nrows, ncols = data_all.shape
    fig, ax = plt.subplots(figsize=(0.5 * nrows, 0.5 * ncols))
    sns.heatmap(data_all, cmap='RdBu_r', center=0, ax=ax)
    ax.set_xticklabels(data_all.columns, rotation=90)
    ax.set_yticks(np.arange(len(data_all.index))+0.5)
    ax.set_yticklabels(data_all.index, rotation=0)
    ax.set_ylabel('Genes')
    ax.set_xlabel('Cell types')
    ax.legend_title = 'Expression level'

    if save:
        if output_name is not None:
            fig.savefig(f'figures/{output_name}', bbox_inches = 'tight')
        else:
            source = adata.obs['source'].values[0]
            if n_top_genes is not None:
                fig.savefig(f'figures/heatmap_top{n_top_genes}_DEG_{source}.pdf', bbox_inches = 'tight')
            else:
                fig.savefig(f'figures/heatmap_allDEG_{source}.pdf', bbox_inches = 'tight')
    return fig

def draw_2d_scatter(adata, save = True, output_name = None):
    x, y = adata.obs['x'].values, adata.obs['y'].values
    labels = adata.obs['labels'].values

    fig, ax = plt.subplots(figsize=(8, 8))
    for i in np.unique(labels):
        ax.scatter(x[labels == i], y[labels == i], s=1.5, label=i, c = np.random.rand(3,))
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    if save:
        if output_name is not None:
            fig.savefig(f'figures/{output_name}', bbox_inches = 'tight')
        else:
            source = adata.obs['source'].values[0]
            fig.savefig(f'figures/single_cell_2d_scatter_{source}.pdf', bbox_inches = 'tight')

    return fig