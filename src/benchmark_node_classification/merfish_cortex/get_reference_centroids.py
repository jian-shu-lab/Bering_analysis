import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

def load_reference():
    path = '/data/aronow/Kang/spatial/Bering/validation/run_bm1/slice_21/output/results_cells_ensembled.h5ad'
    adata = sc.read(path)
    cell_types = np.unique(adata.obs['predicted_labels'].values)
    genes = adata.var.index.values

    reference_centroids = np.zeros((len(cell_types), len(genes)))
    for idx, cell_type in enumerate(cell_types):
        adata_type = adata[adata.obs['predicted_labels'] == cell_type].copy()
        reference_centroids[idx, :] = adata_type.X.mean(axis=0)
    
    return reference_centroids, cell_types, genes

reference_centroids, cell_types, genes = load_reference()
with open('reference_centroids.pkl', 'wb') as f:
    pickle.dump(reference_centroids, f)
with open('cell_types.pkl', 'wb') as f:
    pickle.dump(cell_types, f)
with open('genes.pkl', 'wb') as f:
    pickle.dump(genes, f)