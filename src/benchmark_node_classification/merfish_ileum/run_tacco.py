# https://simonwm.github.io/tacco/notebooks/osmFISH_single_molecule_annotation.html

import os
import sys
import json
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import sklearn.metrics
from anndata import AnnData
from scipy.sparse import csr_matrix

import tacco as tc

def load_spots_table():
    path = '/data/aronow/Kang/spatial/Bering/ileum/data/merfish_illeum_all_v2.tsv'
    df_spots = pd.read_csv(path, sep = '\t', header = 0, index_col = 0)
    return df_spots

def load_references():
    # path = '/data/aronow/Kang/spatial/Bering/validation/run_bm1/slice_21/output/results_cells_ensembled.h5ad'
    # adata = sc.read(path)
    # 5801 x 241
    df_segmented = pd.read_csv('/data/aronow/Kang/spatial/data/MERFISH/Petukhov_et_al_2022_NatBiotechnology/data_release_baysor_merfish_gut/data_analysis/baysor/segmentation/segmentation.csv',header=0,index_col=0)
    df_segmented = df_segmented[df_segmented['cell'] != 0].copy()
    df_cluster = pd.read_csv('/data/aronow/Kang/spatial/data/MERFISH/Petukhov_et_al_2022_NatBiotechnology/data_release_baysor_merfish_gut/data_analysis/baysor/clustering/cell_assignment.csv', header=0, index_col=0)
    df_cluster.index = df_cluster.index.astype(str)
    df_expr = df_segmented.groupby(['cell','gene']).size().unstack(fill_value=0)
    adata = AnnData(
        X = csr_matrix(df_expr.values),
        obs = df_expr.index.to_frame(),
        var = df_expr.columns.to_frame()
    )
    adata.obs['labels'] = df_cluster.loc[adata.obs.index.values, 'leiden_final'].values

    return adata

if __name__ == '__main__':
    
    molecules = load_spots_table()
    molecules.rename({'features': 'gene'}, axis = 1, inplace = True)
    print(molecules.head())

    reference = load_references()
    print(reference)

    result = tc.tl.annotate_single_molecules(molecules, reference, annotation_key='labels', method='projection')
    molecules['tacco_labels'] = result

    fig, ax = plt.subplots(figsize=(10, 10))
    x, y = molecules['x'].values, molecules['y'].values
    for label in np.unique(result):
        ax.scatter(x[result == label], y[result == label], s = 0.01, label = label, c = np.random.rand(3,))

    molecules['tacco_labels'] = result
    molecules.to_csv('result_tacco.tsv', sep = '\t')
    ax.legend(markerscale = 18, fontsize = 10)
    ax.set_title('tacco')
    fig.savefig('tacco.png', dpi = 300)

    # molecules = pd.read_csv('result_tacco.tsv', sep = '\t', header = 0, index_col = 0)
    # labels = molecules['tacco_labels'].values
    # fig, ax = plt.subplots(figsize=(10, 10))
    # x, y = molecules['x'].values, molecules['y'].values
    # for label in np.unique(labels):
    #     ax.scatter(x[labels == label], y[labels == label], s = 0.01, label = label, c = cmap[label])

    # ax.legend(markerscale = 28, fontsize = 14)
    # # no axis
    # ax.set_xticks([])
    # ax.set_yticks([])

    # ax.set_title('Tacco prediction')
    # fig.savefig('tacco_v2.png', dpi = 300)