import pickle
import logging
import numpy as np
import pandas as pd

from anndata import AnnData

import tifffile as tiff
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from ClusterMap.clustermap import *
logger = logging.getLogger(__name__)

def run_clustermap(
    df_spots, img, gene_list, minx = None, maxx = None, miny = None, maxy = None, 
    dapi_grid_interval=5, LOF=False, contamination=0.1, pct_filter=0.1, cell_num_threshold=0.01,
    add_dapi=True,use_genedis=True
):
    # 1. load spots
    df_spots['gene'] = df_spots['features'].values # dummy column
    df_spots = df_spots.loc[:, ['x', 'y', 'gene', 'segmented', 'features', 'labels']]
    df_spots['x'] = df_spots['x'].astype(np.int16)
    df_spots['y'] = df_spots['y'].astype(np.int16)
    df_spots.columns = ['spot_location_1', 'spot_location_2', 'gene', 'cell', 'gene_name', 'labels']

    # 1.2 subset to save time
    if (minx is not None) and (maxx is not None) and (miny is not None) and (maxy is not None):
        img = img[miny:maxy, minx:maxx]
        logger.info(f'Original spots table shape: {df_spots.shape}')
        df_spots = df_spots.loc[(df_spots['spot_location_1']> minx) & ((df_spots['spot_location_1'] < maxx)),:].copy()
        df_spots = df_spots.loc[(df_spots['spot_location_2']> miny) & ((df_spots['spot_location_2'] < maxy)),:].copy()

        df_spots['spot_location_1'] = df_spots['spot_location_1'] - minx
        df_spots['spot_location_2'] = df_spots['spot_location_2'] - miny

    gene_name_dict = dict(zip(df_spots['gene_name'].unique(), np.arange(1, 1+len(df_spots['gene_name'].unique()))))
    df_spots['gene'] = [gene_name_dict[i] for i in df_spots['gene_name'].values]

    df_spots['index'] = np.arange(0, df_spots.shape[0])
    df_spots.set_index('index', inplace = True)

    logger.info(f'Spots table shape after subsample: {df_spots.shape}')

    gene_list = df_spots['gene'].unique()

    # 2. build model
    logger.info(f'Running ClusterMap')
    model = ClusterMap(
        spots = df_spots,
        dapi = img,
        gene_list = gene_list,
        num_dims = 2,
        xy_radius = 40,
        z_radius = 0,
        fast_preprocess = True,
        gauss_blur = True,
        sigma = 1,
    )
    with open('output/model_clustermap_step1.pl', 'wb') as p:
        pickle.dump(model, p)

    # 4. preprocess data
    logger.info(f'Running ClusterMap - preprocess')
    model.preprocess(dapi_grid_interval=dapi_grid_interval, LOF=LOF, contamination=contamination, pct_filter=pct_filter)

    # 5. cell segmentation
    logger.info(f'Running ClusterMap - segmentation')
    model.segmentation(cell_num_threshold=cell_num_threshold, dapi_grid_interval=dapi_grid_interval, add_dapi=add_dapi,use_genedis=use_genedis)

    with open('output/model_clustermap_step2.pl', 'wb') as p:
        pickle.dump(model, p)
    model.spots.to_csv('output/clustermap_result.txt', sep = '\t')

    # 6. plot segmentation
    model.plot_segmentation(figsize=(8,8),s=0.005,plot_with_dapi=True,plot_dapi=True,savepath='figures/clustermap.png')