import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.metrics import mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

from ._settings import STAT_KEYS as K

# BERING_CELL_COLUMN = 'predicted_cells'
BERING_CELL_COLUMN = 'ensembled_cells'

def _get_num_cells(df_results = None, mask = None, method = 'Bering'):
    '''
    Number of cells in Bering results
    '''
    print(method)
    cell_id_col = K.CELLID_COLNAME_DICT[method]
    if cell_id_col is None:
        if mask is None:
            raise ValueError('Mask must be specified for Cellpose or Watershed')
        else:
            num_cells = len(np.unique(mask)) - 1 # background is 0
            return num_cells
    else:
        if df_results is None:
            raise ValueError('df_results must be specified for Baysor and other methods')
        num_cells = len(np.unique(df_results[cell_id_col].values)) - 1
        return num_cells

def _get_foreground_acc_prop(df_results = None, mask = None, method = 'Bering'):
    '''
    Watershed or Cellpose requires mask as input
    '''
    print(method)
    groups_original = np.where(df_results['labels'].values == 'background', 'background', 'foreground')
    if (method == 'Watershed' or method == 'Cellpose') and mask is not None:
        x, y = df_results.x.values, df_results.y.values
        x_pixels, y_pixels = x.astype(int), y.astype(int)
        assigned_cells = mask[y_pixels, x_pixels]
        groups_pred = np.where(assigned_cells == 0, 'background', 'foreground')

    elif method in ['Baysor', 'Baysor (no prior)', 'Baysor (w. prior)'] and df_results is not None:
        groups_pred = np.where(df_results['cell'].values == 0, 'background', 'foreground')
    elif method == 'Bering' and df_results is not None:
        groups_pred = np.where(df_results[BERING_CELL_COLUMN].values == -1, 'background', 'foreground')
    elif method in ['ClusterMap', 'ClusterMap (no image)', 'ClusterMap (w. image)'] and df_results is not None:
        groups_pred = np.where(df_results['clustermap']==-1, 'background', 'foreground')
    elif method in ['Raw', 'Original', 'Paper']:
        groups_pred = groups_original

    accuracy = np.sum(groups_original == groups_pred) / len(groups_original)
    proportion_foreground = np.sum(groups_pred == 'foreground') / len(groups_pred)
    return accuracy, proportion_foreground

def _accuracy_cell_types(df_results):
    '''
    Accuracy of cell types in Bering results
    '''
    df_results = df_results[df_results['labels'] != 'background'].copy() # only foreground spots
    cell_types = df_results['labels'].values
    cell_types_pred = df_results['predicted_node_labels'].values
    cell_types_predEnsem = df_results['ensembled_labels'].values

    accuracy_pred = np.sum(cell_types == cell_types_pred) / len(cell_types)
    accuracy_predEnsemble = np.sum(cell_types == cell_types_predEnsem) / len(cell_types)
    return accuracy_pred, accuracy_predEnsemble


def _get_cells(df_results, method = 'Bering', keep_background = False):
    # get foreground spots
    if not keep_background:
        df_results = df_results[df_results['labels'] != 'background'].copy()

    # get cells
    if method in ['ClusterMap', 'ClusterMap (no image)', 'ClusterMap (w. image)']:
        if 'cell_original' in df_results.columns:
            cells_original = df_results['cell_original'].values
        else:
            cells_original = df_results['cell'].values
        cells_segmented = df_results['clustermap'].values
    elif method in ['Bering', 'Baysor', 'Baysor (no prior)', 'Baysor (w. prior)']:
        cells_original = df_results['segmented'].values
        if method in ['Baysor', 'Baysor (no prior)', 'Baysor (w. prior)']:
            cells_segmented = df_results['cell'].values
        elif method == 'Bering':
            cells_segmented = df_results[BERING_CELL_COLUMN].values
    return cells_original, cells_segmented


def _get_mi_ari(df_results = None, mask = None, method = 'Bering', keep_background = False, subset_ratio = None):
    '''
    Get mutual information between segmentation and spot detection
    '''
    print(method)
    if subset_ratio is not None:
        np.random.seed(0)
        num_rows = int(len(df_results) * subset_ratio)
        df_results = df_results.iloc[np.random.choice(len(df_results), num_rows, replace = False), :]
    if method in ['Watershed', 'Cellpose'] and mask is not None:
        # df_results for watershed and cellpose are actually df_spots (raw spots)
        if not keep_background:
            df_results = df_results[df_results['labels'] != 'background'].copy()
        x, y = df_results.x.values, df_results.y.values
        cells_original = df_results.segmented.values

        x_pixels, y_pixels = x.astype(int), y.astype(int)
        cells_segmented = mask[y_pixels, x_pixels]
    else:
        cells_original, cells_segmented = _get_cells(df_results, method = method, keep_background = keep_background)

    # get mutual information
    # mi = mutual_info_score(cells_original, cells_segmented)
    mi = adjusted_mutual_info_score(cells_original, cells_segmented)
    # ari = adjusted_rand_score(cells_original, cells_segmented)
    ari = 0.0

    import pickle
    with open(f'output/cells_original_{method}.pl', 'wb') as p:
        pickle.dump(cells_original, p)

    with open(f'output/cells_segmented_{method}.pl', 'wb') as p:
        pickle.dump(cells_segmented, p)

    return mi, ari

def _get_cell_area(df_results = None, mask = None, method = 'Bering', strategy = 'pixels'):
    '''
    Get cell area
    '''
    xcol = 'x'
    ycol = 'y'
    if method in ['Watershed', 'Cellpose'] and mask is not None:
        # df_results for watershed and cellpose are actually df_spots (raw spots)
        x, y = df_results.x.values, df_results.y.values
        x_pixels, y_pixels = x.astype(int), y.astype(int)
        assigned_cells = mask[y_pixels, x_pixels]
        df_results['cell'] = assigned_cells
        # df_results = df_results[df_results['labels'] != 'background'].copy()
        df_results = df_results[df_results['cell'] != 0].copy()
    elif method in ['Baysor', 'Baysor (no prior)', 'Baysor (w. prior)']:
        df_results = df_results[df_results['cell'] != 0].copy()
    elif method == 'Bering':
        df_results = df_results[df_results[BERING_CELL_COLUMN] != -1].copy()
        df_results['cell'] = df_results[BERING_CELL_COLUMN]
    elif method in ['ClusterMap', 'ClusterMap (no image)', 'ClusterMap (w. image)']:
        df_results = df_results[df_results['clustermap'] != -1].copy()
        df_results['cell'] = df_results['clustermap']
        xcol = 'spot_location_1'
        ycol = 'spot_location_2'
    elif method in ['Raw', 'Original', 'Paper']:
        df_results = df_results[df_results['labels'] != 'background'].copy()
        df_results['cell'] = df_results['segmented']

    if strategy == 'convex':
        unique_cells = np.unique(df_results['cell'].values)
        avail_cells = []
        cells = df_results['cell'].values
        cell_area = []
        for cell in unique_cells:
            # create convex hull and calculate area
            x = df_results.loc[cells == cell, xcol].values
            y = df_results.loc[cells == cell, ycol].values
            if len(x) > 3 and (x.min() != x.max()) and (y.min() != y.max()):
                avail_cells.append(cell)
                points = np.array([x, y]).T
                hull = ConvexHull(points)
                area = hull.area
                cell_area.append(area)
            else:
                continue
    elif strategy == 'pixels':
        df_results[xcol] = df_results[xcol].astype(int)
        df_results[ycol] = df_results[ycol].astype(int)

        df_results['coords'] = [(i,j) for i,j in zip(df_results[xcol].values, df_results[ycol].values)]
        df_coords = df_results.groupby('cell')['coords'].apply(list).reset_index()
        df_coords['area'] = df_coords['coords'].apply(lambda x: len(x))
        avail_cells = df_coords['cell'].values
        cell_area = df_coords['area'].values

    return avail_cells, cell_area

def _get_num_tps_percell(df_results = None, mask = None, method = 'Bering'):
    '''
    Get cell count matrix
    '''
    if method in ['Watershed', 'Cellpose'] and mask is not None:
        # df_results for watershed and cellpose are actually df_spots (raw spots)
        x, y = df_results.x.values, df_results.y.values
        x_pixels, y_pixels = x.astype(int), y.astype(int)
        assigned_cells = mask[y_pixels, x_pixels]
        df_results['cell'] = assigned_cells
        df_results = df_results[df_results['cell'] != 0].copy()
    elif method in ['Baysor', 'Baysor (no prior)', 'Baysor (w. prior)']:
        df_results = df_results[df_results['cell'] != 0].copy()
    elif method == 'Bering':
        df_results = df_results[df_results[BERING_CELL_COLUMN] != -1].copy()
        df_results['cell'] = df_results[BERING_CELL_COLUMN]
    elif method in ['ClusterMap', 'ClusterMap (no image)', 'ClusterMap (w. image)']:
        df_results = df_results[df_results['clustermap'] != -1].copy()
        df_results['cell'] = df_results['clustermap']
    elif method in ['Raw', 'Original', 'Paper']:
        df_results = df_results[df_results['labels'] != 'background'].copy()
        df_results['cell'] = df_results['segmented']

    # unique_cells = np.unique(df_results['cell'].values)
    # cell_n_tps = []
    # for cell in unique_cells:
    #     n_tps = len(df_results.loc[df_results['cell'] == cell, :])
    #     cell_n_tps.append(n_tps)
    df_abundance = pd.DataFrame(df_results.groupby('cell').size(), columns = ['n_tps'])
    print(f'Number of cells in method {method}: {len(df_abundance)}')
    unique_cells = df_abundance.index.values
    cell_n_tps = df_abundance['n_tps'].values

    return unique_cells, cell_n_tps
