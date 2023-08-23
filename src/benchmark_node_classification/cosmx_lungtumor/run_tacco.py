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
    fov = 10
    path = f'/data/aronow/Kang/spatial/Bering/cosmx/data/transcripts_aligned_celltypes/Lung5_Rep1_tx_fov_{fov}.txt'
    df_spots = pd.read_csv(path, sep = '\t', header = 0, index_col = 0)
    return df_spots

def load_references():
    fov = 10
    df_spots_all = pd.read_csv(f'/data/aronow/Kang/spatial/Bering/cosmx/data/transcripts_aligned_celltypes/Lung5_Rep1_tx_fov_{fov}.txt', sep = '\t', header = 0, index_col = 0)
    df_spots_seg = df_spots_all.loc[df_spots_all['labels'] != 'background', :].copy()
    df_spots_seg['segmented'] = df_spots_seg['segmented'].astype(str)

    df_expr = df_spots_seg.groupby(['segmented','features']).size().unstack(fill_value=0)
    adata = AnnData(
        X = csr_matrix(df_expr.values),
        obs = df_expr.index.to_frame(),
        var = df_expr.columns.to_frame()
    )
    # get first label for each cell as the label of the cell
    labels_dict = df_spots_seg.groupby(['segmented'])['labels'].first()
    adata.obs['labels'] = labels_dict.loc[adata.obs.index.values].values
    df_types = pd.read_csv('/data/aronow/Kang/spatial/Bering/cosmx/data/celltypes_simplification.txt', sep = '\t', header = 0, index_col = 0)
    adata.obs['labels'] = df_types.loc[adata.obs.labels.values, 'new'].values

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