import random
import numpy as np
import pandas as pd 
import multiprocessing as mp
from tqdm import tqdm

from anndata import AnnData
from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors

import matplotlib as mpl
import matplotlib.pyplot as plt

def load_data(FOV = 5):
    PATH = '/data/aronow/Kang/spatial/Bering/cosmx/data/'
    df_spots = pd.read_csv(PATH + 'transcripts_aligned_celltypes/Lung5_Rep1_tx_fov_' + str(FOV) + '.txt', sep = '\t', header = 0, index_col = 0)
    print(df_spots.head())
    print(df_spots.columns)

    return df_spots

def get_ncvs(df_spots_all, cell_id, compartments, genes, n_graphs = 1, n_neighbors = 10, min_spots = 15):
    
    df_counts_all = pd.DataFrame(index = genes)
    for compartment in compartments:
        print(f'Cell ID: {cell_id}, Compartment: {compartment}')
        df_counts = pd.DataFrame(index = genes)
        df_spots = df_spots_all.loc[(df_spots_all['segmented'] == cell_id) & (df_spots_all['components'] == compartment), :].copy()
        
        if df_spots.shape[0] < np.max([min_spots, n_neighbors + 1]):
            continue

        label = df_spots.labels.values[0]
        indices = np.arange(df_spots.shape[0])
        x = df_spots.x.values
        y = df_spots.y.values
        features = df_spots.features.values
        coords = np.array([x, y]).T
        nbrs = NearestNeighbors(n_neighbors = n_neighbors).fit(coords)

        selected_indices = random.sample(list(indices), n_graphs)
        
        for idx in selected_indices:
            xi = x[idx]
            yi = y[idx]
            
            _, indices = nbrs.kneighbors(np.array([[xi, yi]]))
            features_neigh = features[indices]

            features_avail, features_cnts = np.unique(features_neigh, return_counts = True)
            colname = str(cell_id) + '_' + compartment + '_' + label + '_' + str(idx)
            print(colname)
            df_counts[colname] = 0
            df_counts.loc[features_avail, colname] = features_cnts

        df_counts_all = pd.concat([df_counts_all, df_counts], axis = 1)
    print(df_counts_all)

    return df_counts_all

if __name__ == '__main__':
    df_spots = load_data()

    genes = df_spots.features.unique()
    cells = df_spots.segmented.unique()
    cells = np.random.choice(cells, 2048, replace = False)
    Counts = pd.DataFrame(index = genes)
    
    # n_neighbors_list = [5, 10, 15, 20, 25, 30]
    n_neighbors_list = [20, 25, 30]
    compartments = ['Cytoplasm', 'Membrane', 'Nuclear']

    for n_neighbors in n_neighbors_list:
        Counts = []

        pool = mp.Pool(processes = 32)
        out = [pool.apply_async(get_ncvs, args = (df_spots, cell, compartments, genes, 2, n_neighbors)) for cell in cells]
        pool.close()
        pool.join()

        for collect in out:
            Counts.append(collect.get())

        Counts = pd.concat(Counts, axis = 1)
        Counts = Counts.T 
        adata = AnnData(
            X = csr_matrix(Counts.values),
            obs = pd.DataFrame(index = Counts.index.values),
            var = pd.DataFrame(index = Counts.columns)
        )
        print(adata.shape)
        adata.write(f'compartment_data/NCV_all_cell_compartments_n_cells_{len(cells)}_n_neighbors_{n_neighbors}.h5ad')