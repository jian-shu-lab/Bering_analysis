import numpy as np
import pandas as pd
import scanpy as sc

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt 

def _format_DEGs(adata):
    keys = ["names","scores","logfoldchanges","pvals","pvals_adj","pts"]
    for i,key in enumerate(keys):
        a = pd.DataFrame(adata.uns["rank_genes_groups"][key]) # transfer to data frame
        b = pd.DataFrame(a.values.T.reshape(1,a.shape[0]*a.shape[1]).T) # reformat the data frame to one column
           
        if i == 0:
            b.columns = [key] # rename the column name
            b["Status"] = sorted(list(a.columns)*a.shape[0]) # add Status annotation
            b.set_index([key],inplace=True)
            b_merged = b
        else:
            if key in ["pts"]:
                pts_all = []
                for cell_group in np.unique(b_merged["Status"]):
                    genes = b_merged.loc[b_merged["Status"] == cell_group,:].index.values
                    pts_all = pts_all + list(a.loc[genes, cell_group])
                b_merged[key] = pts_all
            else:
                b_merged[key] = list(b[0])
        
    return b_merged

def get_DEGs(adata, groupby = 'labels', groups = 'all', reference = 'rest', method = 'wilcoxon', min_cells_per_group = 3):
    '''
    input adata should be raw datae here
    '''
    adata = adata.copy()
    labels_abundances = pd.DataFrame(adata.obs[groupby].value_counts())
    labels_abundances.columns = ['abundance']
    avail_labels = labels_abundances[labels_abundances['abundance'] >= min_cells_per_group].index
    print(adata.obs[groupby].value_counts())
    adata = adata[adata.obs[groupby].isin(avail_labels), :].copy()
    print(adata.obs[groupby].value_counts())
    adata.obs[groupby] = adata.obs[groupby].astype('category').cat.set_categories(avail_labels)
    print(adata.obs[groupby].cat.categories)

    print(adata)
    sc.pp.normalize_total(adata, target_sum = 1000)
    sc.pp.log1p(adata, base = 2)

    sc.tl.rank_genes_groups(
        adata, 
        groupby = groupby, 
        groups = groups, 
        reference = reference,
        method = method, 
        n_genes = adata.shape[1], 
        pts = True, 
        # use_raw = True,
        use_raw = False,
    )
    df_DEG = _format_DEGs(adata)

    # expr_avg = pd.DataFrame(index = adata.var_names, columns = adata.obs[groupby].cat.categories)
    # for i in adata.obs[groupby].cat.categories:
    #     expr_avg[i] = adata[adata.obs[groupby] == i].raw.X.mean(axis = 0).A1
    
    # for i in adata.obs[groupby].cat.categories:
    #     df_DEG[i] = expr_avg.loc[df_DEG.index, i]

    return df_DEG