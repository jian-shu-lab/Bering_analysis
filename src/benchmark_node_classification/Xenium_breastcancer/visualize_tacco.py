# https://simonwm.github.io/tacco/notebooks/osmFISH_single_molecule_annotation.html

import os
import sys
import json
import pickle
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

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 18

def load_results():
    df_tacco = pd.read_csv('result_tacco.tsv', sep = '\t', header = 0, index_col = 0, keep_default_na=False)
    df_tacco.loc[df_tacco['labels'].isin(['']), 'labels'] = 'background'

    df_bering_baseline = pd.read_csv('/data/aronow/Kang/spatial/Bering/validation/run_xenium/v1/baseline/output/results.txt', sep = '\t', header = 0, index_col = 0)
    df_bering = pd.read_csv('/data/aronow/Kang/spatial/Bering/validation/run_xenium/v1/output/results.txt', sep = '\t', header = 0, index_col = 0)

    return df_tacco, df_bering_baseline, df_bering

if __name__ == '__main__':
    df_tacco, df_bering_baseline, df_bering = load_results()

    # tacco
    x, y = df_tacco['x'].values, df_tacco['y'].values
    labels_true = df_tacco['labels'].values
    labels_pred = df_tacco['tacco_labels'].values
    np.random.seed(42)
    label_col_dict = {label:np.random.rand(3,) for label in np.unique(labels_true)}
    with open('label_col_dict.pkl', 'wb') as f:
        pickle.dump(label_col_dict, f)

    # narrow gaps between subplots
    fig, axes = plt.subplots(figsize = (20, 20), nrows = 2, ncols = 2, sharex = True, sharey = True, gridspec_kw = {'wspace': 0.05, 'hspace': 0.05})
    for label in np.unique(labels_true):
        if label in ['background', 'Unlabeled']:
            axes[0,0].scatter(x[labels_true == label], y[labels_true == label], color = '#DCDCDC', label = label, s = 0.005, alpha = 0.3)
        else:
            axes[0,0].scatter(x[labels_true == label], y[labels_true == label], color = label_col_dict[label], label = label, s = 0.005)
    # axes[0].set_title('Original labels', fontsize = 40)
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])
    for label in np.unique(labels_pred):
        axes[0,1].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = 0.005)
    # axes[1].set_title('Tacco predicted labels', fontsize = 40)
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])
    x, y = df_bering['x'].values, df_bering['y'].values

    # bering baseline
    labels_pred = df_bering_baseline['predicted_node_labels'].values
    for label in np.unique(labels_pred):
        if label in ['background', 'Unlabeled']:
            axes[1,0].scatter(x[labels_pred == label], y[labels_pred == label], color = '#DCDCDC', label = label, s = 0.005, alpha = 0.3)
        else:
            axes[1,0].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = 0.005)
    # axes[2].set_title('Bering (baseline) predicted labels', fontsize = 40)
    axes[1,0].set_xticks([])
    axes[1,0].set_yticks([])    

    # bering
    labels_pred = df_bering['predicted_node_labels'].values
    for label in np.unique(labels_pred):
        if label in ['background', 'Unlabeled']:
            axes[1,1].scatter(x[labels_pred == label], y[labels_pred == label], color = '#DCDCDC', label = label, s = 0.005, alpha = 0.3)
        else:
            axes[1,1].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = 0.005)
    # axes[3].set_title('Bering predicted labels', fontsize = 40)
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])
    # add legend on the right side of the plot
    plt.legend(fontsize = 15, markerscale = 60, bbox_to_anchor = (1.15, 1), loc = 2, borderaxespad = 0.)

    plt.savefig('tacco_bering_v3.png', dpi = 300, bbox_inches = 'tight')