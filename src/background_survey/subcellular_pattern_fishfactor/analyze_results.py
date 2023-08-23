import sys
import pickle
import numpy as np
import pandas as pd
import torch 

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append('/data/aronow/Kang/spatial/Bering/postTalk_background/subcellular_patterns/FISHFactor_master')
import src as src
from src.simulation import simulate_data
from src.fishfactor import FISHFactor

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 12

def load_data():
    # with open('w.pkl', 'rb') as f:
    #     w = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # return model, w
    return model

def plot_weights(df_weights, factor_score_thresh = 0.9):
    
    weights = np.max(df_weights.values, axis=1)
    df_weights = df_weights[weights > factor_score_thresh]
    df_weights = df_weights.T.copy()
    
    sns.clustermap(df_weights, cmap='vlag', figsize=(18, 2.5))
    plt.savefig('figures/weight_matrix_v2.pdf', dpi=300, bbox_inches='tight')

def plot_factors(model, n_cells = 10, n_factors = 3):
    z = model.get_factors() # shape (n_cells, n_factors, n_y, n_x)

    fig, axes = plt.subplots(n_factors, n_cells, figsize=(n_cells*3, 3 * n_factors))

    for cell in range(n_cells):
        for k in range(z.shape[1]):
            im = axes[k, cell].matshow(z[cell, k])
            axes[k, cell].axis('off')
            plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    # set title
    for cell in range(n_cells):
        axes[0, cell].set_title(f'Cell {cell}', fontsize=12)
    for k in range(n_factors):
        axes[k, 0].set_ylabel(f'Factor {k}', fontsize=12)

    # remove last column
    axes = axes[:, :-1]

    plt.savefig('figures/factors_v2.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':

    df_weights = pd.read_csv('weights.csv', index_col=0, header=0, sep=',')

    # weight correlation
    f1, f2, f3 = df_weights['Factor 0'].values, df_weights['Factor 1'].values, df_weights['Factor 2'].values
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2))
    axes[0].scatter(f1, f2, s=1); axes[0].set_xlabel('Factor 0'); axes[0].set_ylabel('Factor 1')
    axes[1].scatter(f1, f3, s=1); axes[1].set_xlabel('Factor 0'); axes[1].set_ylabel('Factor 2')
    axes[2].scatter(f2, f3, s=1); axes[2].set_xlabel('Factor 1'); axes[2].set_ylabel('Factor 2')
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('figures/weights_v2.pdf', dpi=300, bbox_inches='tight')
    
    '''
    # weight heatmap
    plot_weights(df_weights)

    # plot factors
    model = load_data()
    plot_factors(model, n_cells = 10, n_factors = 3)
    '''