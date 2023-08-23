import sys
import pickle
import numpy as np
import pandas as pd
import torch 

import matplotlib.pyplot as plt
sys.path.append('/data/aronow/Kang/spatial/Bering/postTalk_background/subcellular_patterns/FISHFactor_master')
import src as src
from src.simulation import simulate_data
from src.fishfactor import FISHFactor

def make_data():
    cell_types = ['tumor 12', 'tumor 5']
    n_cells = 200

    columns = ['x', 'y', 'features', 'segmented']
    columns_new = ['x', 'y', 'gene', 'cell']
    df_spots = pd.read_csv('/data/aronow/Kang/spatial/Bering/benchmark/bm_data/bm2_lungtumor_cosmx_he_et_al/Lung5_Rep1_tx_fov_10.txt',sep='\t',header=0,index_col=0)
    df_spots = df_spots[df_spots['labels'].isin(cell_types)].copy()

    np.random.seed(0)
    random_cells = np.random.choice(df_spots.segmented.unique(), size = n_cells, replace = False)
    df_spots = df_spots[df_spots['segmented'].isin(random_cells)].copy()
    df_spots = df_spots[columns].copy()
    df_spots.columns = columns_new
    return df_spots 

def run_fishfactor(df_spots, n_factors = 5, device = 'cuda:0'):
    print('run fishfactor')
    model = FISHFactor(
        data = df_spots,
        n_factors = n_factors, 
        device = device,
    )
    model.inference(save=False)
    print('done')
    return model 

def plot_loss(model):
    # plot loss
    for cell in range(len(model.losses)):
        plt.plot(torch.arange(len(model.losses[cell])), model.losses[cell], label=f'Cell {cell}')
        plt.legend()
        plt.title('Loss curves for individual cells')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (negative and rescaled ELBO)')
        plt.savefig('figures/loss_curves.png', dpi=300, bbox_inches='tight')

def plot_weights(model):
    w = model.get_weights()
    with open('w.pkl', 'wb') as f:
        pickle.dump(w, f)

    plt.matshow(w.T, cmap='viridis')
    plt.xlabel('Genes')
    plt.ylabel('Factors')
    plt.title('Weight matrix')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.savefig('figures/weight_matrix.png', dpi=300, bbox_inches='tight')

def plot_factors(model, n_cells = 20, n_factors = 5):
    z = model.get_factors() # shape (n_cells, n_factors, n_y, n_x)

    fig, axs = plt.subplots(n_cells, n_factors, figsize=(n_factors*3, 3 * n_cells))

    for cell in range(n_cells):
        for k in range(z.shape[1]):
            im = axs[cell, k].matshow(z[cell, k])
            axs[cell, k].set_title(f'Cell {cell}, Factor {k}')
            axs[cell, k].axis('off')
            plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig('figures/factors.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    print(f'cuda available: {torch.cuda.is_available()}')
    n_factors = 3
    df_spots = make_data()
    model = run_fishfactor(df_spots, n_factors = n_factors)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    plot_loss(model)
    plot_weights(model)
    plot_factors(model, n_factors = n_factors)