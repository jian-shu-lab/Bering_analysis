import pandas as pd
import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

from ._settings import SC_KEYS as K
def _create_anndata(df_results, common_features, mask = None, method = 'Bering'):
    x, y = df_results[K.XCOLS_DICT[method]].values, df_results[K.YCOLS_DICT[method]].values
    features = df_results[K.GENE_COLNAME_DICT[method]].values
    if method in ['Watershed', 'Cellpose']:
        if mask is None:
            raise ValueError('Mask must be specified for Cellpose or Watershed')
        else:
            x, y = df_results.x.values, df_results.y.values
            x_pixels, y_pixels = x.astype(int), y.astype(int)
            cells = mask[y_pixels, x_pixels]
    elif method == 'Raw':
        cells_raw = df_results[K.CELLID_COLNAME_DICT[method]].values
        cells = np.zeros(len(cells_raw),dtype = np.int16)
        labels = df_results['labels'].values
        cells[np.where(labels == 'background')[0]] = -1
        cells[np.where(labels != 'background')[0]] = cells_raw[np.where(labels != 'background')[0]]
    else:
        cells = df_results[K.CELLID_COLNAME_DICT[method]].values

    if method in ['Baysor', 'Baysor (no prior)', 'Baysor (w. prior)', 'Watershed', 'Cellpose']:
        cells[cells==0] = -1 # background is -1
    if method == 'Bering':
        labels = df_results['labels'].values
    
    fg_indices = np.where(cells != -1)[0]
    cells = cells[fg_indices]
    cells = [method + '_' + str(i) for i in cells.astype(str)]
    x, y = x[fg_indices], y[fg_indices]

    features = features[fg_indices]
    if method in ['Raw', 'Bering']:
        labels = labels[fg_indices]
    # Create sparse matrix
    df_expr = pd.DataFrame({'cells': cells, 'features': features})
    df_expr['features'] = df_expr['features'].astype('category')
    df_expr['features'] = df_expr['features'].cat.set_categories(common_features)

    expr_table = df_expr.groupby(['cells', 'features']).size().unstack(fill_value=0)
    adata = AnnData(
        X = csr_matrix(expr_table.values),
        obs = expr_table.index.to_frame(),
        var = expr_table.columns.to_frame(),
    )

    if method in ['Raw', 'Bering']:
        df_expr['labels'] = labels
        df_expr['labels'] = df_expr['labels'].astype('category')
        
        cell_dict = df_expr.groupby(['cells']).first()['labels'].to_frame()
        adata.obs['labels'] = cell_dict.loc[adata.obs.index.values, 'labels'].values
    else:
        adata.obs['labels'] = 'unknown'
    
    cell_coords = pd.DataFrame({'cells': cells, 'x': x, 'y': y})
    cell_coords = pd.DataFrame(cell_coords.groupby(['cells']).median()[['x', 'y']])

    adata.obs['source'] = method
    adata.obs['x'] = cell_coords.loc[adata.obs.index.values, 'x'].values
    adata.obs['y'] = cell_coords.loc[adata.obs.index.values, 'y'].values
    adata.obs['n_counts'] = adata.X.sum(axis = 1)
    adata.obs['n_genes'] = (adata.X > 0).sum(axis = 1)
    adata.obs.set_index('cells', inplace = True)

    return adata

def create_anndata_set(df_results_list, masks_list, method_list, common_features):
    idx = 0
    for df_results, mask, method in zip(df_results_list, masks_list, method_list):
        adata = _create_anndata(df_results, common_features, mask = mask, method = method)
        if idx == 0:
            adata_combined = adata
        else:
            adata_combined = adata_combined.concatenate(adata, index_unique = None, join = 'outer')
        idx = idx + 1

    return adata_combined