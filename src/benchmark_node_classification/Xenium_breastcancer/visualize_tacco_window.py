# https://simonwm.github.io/tacco/notebooks/osmFISH_single_molecule_annotation.html

import os
import sys
import json
import matplotlib as mpl
from matplotlib.patches import Rectangle
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
mpl.rcParams['font.size'] = 22

def load_results():
    df_tacco = pd.read_csv('result_tacco.tsv', sep = '\t', header = 0, index_col = 0, keep_default_na=False)
    df_tacco.loc[df_tacco['labels'].isin(['']), 'labels'] = 'background'

    df_bering_baseline = pd.read_csv('/data/aronow/Kang/spatial/Bering/validation/run_xenium/v1/baseline/output/results.txt', sep = '\t', header = 0, index_col = 0)
    df_bering = pd.read_csv('/data/aronow/Kang/spatial/Bering/validation/run_xenium/v1/output/results.txt', sep = '\t', header = 0, index_col = 0)

    return df_tacco, df_bering_baseline, df_bering

def draw_window(df_tacco, df_bering_baseline, df_bering, label_col_dict, xmin, ymin, window_size = 300, s = 0.3, savename = None):
    df_tacco = df_tacco[(df_tacco['x'] >= xmin) & (df_tacco['x'] <= xmin + window_size) & (df_tacco['y'] >= ymin) & (df_tacco['y'] <= ymin + window_size)].copy()
    df_bering_baseline = df_bering_baseline[(df_bering_baseline['x'] >= xmin) & (df_bering_baseline['x'] <= xmin + window_size) & (df_bering_baseline['y'] >= ymin) & (df_bering_baseline['y'] <= ymin + window_size)].copy()
    df_bering = df_bering[(df_bering['x'] >= xmin) & (df_bering['x'] <= xmin + window_size) & (df_bering['y'] >= ymin) & (df_bering['y'] <= ymin + window_size)].copy()

    # tacco
    x, y = df_tacco['x'].values, df_tacco['y'].values
    labels_true = df_tacco['labels'].values
    labels_pred = df_tacco['tacco_labels'].values
    # label_col_dict = {label:np.random.rand(3,) for label in np.unique(labels_true)}

    # narrow gaps between subplots
    fig, axes = plt.subplots(figsize = (26, 7.5), nrows = 1, ncols = 4, sharex = True, sharey = True, gridspec_kw = {'wspace': 0.05, 'hspace': 0.05})
    for label in np.unique(labels_true):
        if label == 'background':
            axes[0].scatter(x[labels_true == label], y[labels_true == label], color = '#DCDCDC', label = label, s = s, alpha = 0.3)
        else:
            axes[0].scatter(x[labels_true == label], y[labels_true == label], color = label_col_dict[label], label = label, s = s)
    # axes[0].set_title('Original', fontsize = 40)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for label in np.unique(labels_pred):
        axes[1].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = s)
    # axes[1].set_title('Tacco', fontsize = 40)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    x, y = df_bering['x'].values, df_bering['y'].values

    # bering baseline
    labels_pred = df_bering_baseline['predicted_node_labels'].values
    for label in np.unique(labels_pred):
        if label == 'background':
            axes[2].scatter(x[labels_pred == label], y[labels_pred == label], color = '#DCDCDC', label = label, s = s, alpha = 0.3)
        else:
            axes[2].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = s)
    # axes[2].set_title('Bering (baseline)', fontsize = 40)
    axes[2].set_xticks([])
    axes[2].set_yticks([])    

    # bering
    labels_pred = df_bering['predicted_node_labels'].values
    for label in np.unique(labels_pred):
        if label == 'background':
            axes[3].scatter(x[labels_pred == label], y[labels_pred == label], color = '#DCDCDC', label = label, s = s, alpha = 0.3)
        else:
            axes[3].scatter(x[labels_pred == label], y[labels_pred == label], color = label_col_dict[label], label = label, s = s)
    # axes[3].set_title('Bering', fontsize = 40)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    # add legend on the right side of the plot
    # plt.legend(fontsize = 20, markerscale = 60, bbox_to_anchor = (1.15, 1), loc = 2, borderaxespad = 0.)

    if savename is not None:
        plt.savefig(savename, bbox_inches = 'tight', dpi = 300)
    else:
        plt.save(f'window_{xmin}_{ymin}_size_{window_size}.png', bbox_inches = 'tight', dpi = 300)

if __name__ == '__main__':
    df_tacco, df_bering_baseline, df_bering = load_results()
    
    # x (25%, 50%, 75%), y (25%, 50%, 75%)
    # xmin_list = np.percentile(df_tacco['x'].values, [25, 50, 75])
    # ymin_list = np.percentile(df_tacco['y'].values, [25, 50, 75])
    xmin_list = [2300, 2100, 2700]
    ymin_list = [2100, 2500, 2700]
    window_size = 200
    print(xmin_list)
    print(ymin_list)
    
    labels_true = df_tacco['labels'].values
    np.random.seed(0)

    label_col_dict = {
        'B_Cells': '#196985', 
        'CD4+_T_Cells': '#00D6CF', 
        'CD8+_T_Cells': '#7DADEF',
        'DCIS_1': '#68CA00',
        'Endothelial': '#01A6F5',
        'IRF7+_DCs': '#8395BD',
        'LAMP3+_DCs':'#8395BD',
        'Invasive_Tumor': '#F4B0B8', 
        'Macrophages_1': '#E18AC2',
        'Macrophages_2': '#C55E4F',
        'Mast_Cells': '#00A531',
        'Myoepi_ACTA2+': '#006A6A',
        'Myoepi_KRT15+': '#CFABFC', 
        'Perivascular-Like': '#7F37C1',
        'Prolif_Invasive_Tumor': '#E82A5D',
        'Stromal': '#C569D4',
        'Stromal_&_T_Cell_Hybrid': '#F133B8',
        'T_Cell_&_Tumor_Hybrid': '#42CD91',
        'Unlabeled': '#DCDCDC',
        'background': '#DCDCDC',
    }
    # import pickle
    # with open('label_col_dict.pkl', 'rb') as f:
    #     label_col_dict = pickle.load(f)

    # anchors
    x, y, labels_original = df_tacco['x'].values, df_tacco['y'].values, df_tacco['labels'].values
    fig, ax = plt.subplots(figsize = (15, 15))
    for label in np.unique(labels_original):
        if label == 'background':
            ax.scatter(x[labels_original == label], y[labels_original == label], color = '#DCDCDC', label = label, s = 0.05, alpha = 0.3)
        else:
            ax.scatter(x[labels_original == label], y[labels_original == label], color = label_col_dict[label], label = label, s = 0.05)
    for xmin, ymin in zip(xmin_list, ymin_list):
        # draw rectangle
        rect = Rectangle((xmin, ymin), window_size, window_size, linewidth = 3, edgecolor = 'black', facecolor = 'none')
        ax.add_patch(rect)
    fig.savefig('anchors.png', bbox_inches = 'tight', dpi = 300)
                        

    # windows
    for xmin, ymin in zip(xmin_list, ymin_list):
        savename = f'window_{xmin:.2f}_{ymin:.2f}_size_{window_size}_v1.png'
        draw_window(df_tacco, df_bering_baseline, df_bering, label_col_dict, xmin, ymin, window_size, savename = savename)
