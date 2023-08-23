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

def load_results():
    
    df_tacco = pd.read_csv('result_tacco.tsv', sep = '\t', header = 0, index_col = 0)
    # df_tacco = df_tacco[df_tacco['tacco_labels'] != 'Removed'].copy()
    df_bering = pd.read_csv('/data/aronow/Kang/spatial/Bering/validation/run_bm2/ablation_v2/fov10_GNN_rbf_image_noBg/output/results_numEdges_100.txt', sep = '\t', header = 0, index_col = 0)
    df_bering.loc[df_bering['predicted_node_labels'] == 'background_pseudo', 'predicted_node_labels'] = 'background'
    return df_tacco, df_bering

if __name__ == '__main__':
    df_tacco, df_bering = load_results()
    # print(df_tacco.head())
    # print(df_tacco.columns)
    # print(df_tacco.shape)
    # print(df_tacco['tacco_labels'].value_counts())
    # print(df_tacco['labels'].value_counts())
    # print(df_bering.head())
    # print(df_bering.columns)
    # print(df_bering.shape)
    # print(df_bering.predicted_node_labels.value_counts())
    # print(df_bering.labels.value_counts())
    df_types = pd.read_csv('/data/aronow/Kang/spatial/Bering/cosmx/data/celltypes_simplification.txt', sep = '\t', header = 0, index_col = 0)
    df_types.loc['background', :] = ['background']
    df_tacco['labels'] = df_tacco['labels'].map(df_types['new'].to_dict())
    
    # tacco
    x, y = df_tacco['x'].values, df_tacco['y'].values
    labels_true = df_tacco['labels'].values
    labels_pred = df_tacco['tacco_labels'].values
    label_col_dict = {label:np.random.rand(3,) for label in np.unique(labels_true)}
    print(label_col_dict)

    # narrow gaps between subplots
    fig, axes = plt.subplots(figsize = (30, 10), nrows = 1, ncols = 3, sharex = True, sharey = True, gridspec_kw = {'wspace': 0.05, 'hspace': 0.05})
    for label in np.unique(labels_true):
        axes[0].scatter(x[labels_true == label], y[labels_true == label], color = label_col_dict[label], label = label, s = 0.005)
    # axes[0].set_title('Original labels', fontsize = 40)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for label in np.unique(labels_pred):
        axes[1].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = 0.005)
    # axes[1].set_title('Tacco predicted labels', fontsize = 40)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    x, y = df_bering['x'].values, df_bering['y'].values
    labels_pred = df_bering['predicted_node_labels'].values
    for label in np.unique(labels_pred):
        axes[2].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = 0.005)
    # axes[2].set_title('Bering predicted labels', fontsize = 40)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    # add legend on the right side of the plot
    plt.legend(fontsize = 15, markerscale = 90, bbox_to_anchor = (1.15, 1), loc = 2, borderaxespad = 0.)

    plt.savefig('tacco_bering_v3.png', dpi = 300, bbox_inches = 'tight')