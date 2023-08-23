import scanpy as sc

def run_scanpy(adata, method = None, min_counts = 10, target_sum = 1e3, n_neighbors = 10, save = True):
    sc.pp.filter_cells(adata, min_counts = min_counts)
    sc.pp.normalize_total(adata, target_sum = target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata
    
    sc.pp.scale(adata)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors = n_neighbors)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    if method is None:
        suffix = ''
    else:
        suffix = method
    if 'labels' in adata.obs.columns:
        sc.pl.umap(adata, color = 'labels', save = f'_labels_{suffix}.png')
    sc.pl.umap(adata, color = 'source', save = f'_source_{suffix}.png')
    sc.pl.umap(adata, color = ['n_counts','n_genes'], save = f'_dataQC_{suffix}.png')
    
    return adata