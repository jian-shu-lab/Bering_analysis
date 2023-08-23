import os
import sys
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile as tiff

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 16

sys.path.append('/data/aronow/Kang/spatial/Bering/benchmark/bm_code/')
import src as bm

markers_nontumor = ['CCR7', 'CD14', 'CD53', 'CD68', 'CD74', 'CD163', 'COL3A1', 'COL4A1', 'COL4A2', 'COL5A1', 'COL6A3', 'SOD2', 'SRGN', 'TAGLN', 'VIM']
markers_tumor = ['CEACAM6', 'DDR1', 'EPCAM', 'EPHA2', 'ITGB4', 'KRT8', 'KRT17', 'KRT19', 'S100P', 'SLC2A1', 'TM4SF1']

def draw_gene_umap(adata, gene_list, savename):
    sc.pl.umap(adata, color = gene_list, cmap = 'Blues', ncols = 5, frameon = False, show = False, save = savename, vmax = 5)

if __name__ == '__main__':

    # 1. survey in single cell data
    combined = sc.read('../adata_combined_raw.h5ad')
    sources = ['Raw', 'Watershed', 'Cellpose', 'ClusterMap (no image)', 'ClusterMap (w. image)', 'Baysor (no prior)', 'Baysor (w. prior)', 'Bering']

    for source in sources:

        raw = combined[combined.obs.source == source, :].copy()
        sc.pp.filter_cells(raw, min_counts = 10)
        sc.pp.normalize_total(raw, target_sum = 1e3)
        
        processed = sc.read(f'adata_processed_{source}_transfer_simplified.h5ad')
        raw = raw[processed.obs.index.values, :].copy()
        processed.X = raw.X.copy()

        savename = f'_markers_tumor_genes_{source}.png'
        draw_gene_umap(processed, gene_list = markers_tumor, savename = savename)

        savename = f'_markers_nontumor_genes_{source}.png'
        draw_gene_umap(processed, gene_list = markers_nontumor, savename = savename)