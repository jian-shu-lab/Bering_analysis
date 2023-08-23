import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ssam

### 1. data prepration
# df = pd.read_csv(
#     "zenodo/multiplexed_smFISH/raw_data/smFISH_MCT_CZI_Panel_0_spot_table.csv",
#     usecols=['x', 'y', 'z', 'target'])

# reference_centroids, ref_cell_types, ref_genes = load_reference()

def save_celltype_maps(df, celltype_maps, colname):
    x, y = df.x.values.astype(int), df.y.values.astype(int)
    prediction = celltype_maps[x, y]
    df[colname] = prediction
    return df

### 1. load reference data
with open('reference_centroids.pkl', 'rb') as f:
    reference_centroids = pickle.load(f)
with open('cell_types.pkl', 'rb') as f:
    ref_cell_types = pickle.load(f)
with open('genes.pkl', 'rb') as f:
    ref_genes = pickle.load(f)


### 2. prepare query data
path = '/data/aronow/Kang/spatial/Bering/validation/run_bm1/slice_21/mouse1_slice21_spots_all_v2.tsv'
df = pd.read_csv(path, sep = '\t', header = 0, index_col = 0)
df.rename({'features': 'target'}, axis = 1, inplace = True)
df = df.loc[:, ['x','y','z','target']]

# um_per_pixel = 0.1
um_per_pixel = 1.0

df.x = (df.x - df.x.min()) * um_per_pixel + 10
df.y = (df.y - df.y.min()) * um_per_pixel + 10
df.z = (df.z - df.z.min()) * um_per_pixel + 10
width = df.x.max() - df.x.min() + 10
height = df.y.max() - df.y.min() + 10

grouped = df.groupby('target').agg(list)
genes = list(grouped.index)
coord_list = []
for target, coords in grouped.iterrows():
    coord_list.append(np.array(list(zip(*coords))))

print(len(ref_genes), len(genes), len(np.intersect1d(ref_genes, genes)))

### 3. create data object
ds = ssam.SSAMDataset(genes, coord_list, width, height)
analysis = ssam.SSAMAnalysis(
  ds,
  ncores=10, # used for kde step
  save_dir="kde/",
  verbose=True)

### 4. run ssam
bd = 1.0
analysis.run_kde(bandwidth=bd, use_mmap=False)
print(f' --- run_kde done, vf shape: {analysis.dataset.vf.shape}')

analysis.find_localmax(
    search_size=3,
    # min_norm = 0.05,
    # min_expression = 0.005,
    min_norm=0.2,
    min_expression=0.027
)
print(' --- find_localmax done, vf shape: ', analysis.dataset.vf.shape)

analysis.normalize_vectors_sctransform()
print(' --- normalize_vectors_sctransform done, vf shape: ', analysis.dataset.vf.shape)

### 5. create guided cell map
analysis.cluster_vectors(
    min_cluster_size=0,
    pca_dims=22,
    resolution=0.15,
    metric='correlation')
print(' --- cluster_vectors done, vf shape: ', analysis.dataset.vf.shape)

analysis.map_celltypes(centroids = reference_centroids)
print(' --- map_celltypes done, vf shape: ', analysis.dataset.vf.shape)
print(f'cell type matches shape: {analysis.dataset.celltype_maps.shape}')

celltype_maps = analysis.dataset.celltype_maps[:,:,0]
max_coor = analysis.dataset.max_correlations[:,:,0]
save_celltype_maps(df, celltype_maps, colname = 'ssam_unfiltered')
np.save(f'ssam_celltype_maps_um_per_pixel={um_per_pixel}_bd={bd}.npy', celltype_maps)
np.save(f'ssam_max_correlations_um_per_pixel={um_per_pixel}_bd={bd}.npy', max_coor)

fig, ax = plt.subplots(figsize=[10, 10])
for idx, cell_type in enumerate(ref_cell_types):
    cell_type_indices = np.where(celltype_maps[df.x.values.astype(int), df.y.values.astype(int)] == idx)[0]
    ax.scatter(df.x.values[cell_type_indices], df.y.values[cell_type_indices], s = 0.015, label = cell_type)
    
ax.legend(markerscale = 15, fontsize = 15)
plt.savefig(f'ssam_celltype_map_um_per_pixel={um_per_pixel}.png', dpi = 300)

### 6. filter cell type maps
filter_method = "local"
filter_params = {
    "block_size": 151,
    "method": "mean",
    "mode": "constant",
    "offset": 0.2
}
analysis.filter_celltypemaps(
    min_norm=filter_method, 
    filter_params=filter_params, 
    min_r=0.3, fill_blobs=True, min_blob_area=50, 
    output_mask=None,
)
celltype_maps = analysis.dataset.celltype_maps[:,:,0]
max_coors = analysis.dataset.max_correlations[:,:,0]
np.save(f'ssam_celltype_maps_filtered_um_per_pixel={um_per_pixel}_bd={bd}.npy', celltype_maps)
np.save(f'ssam_max_correlations_filtered_um_per_pixel={um_per_pixel}_bd={bd}.npy', max_coors)

save_celltype_maps(df, celltype_maps, colname = 'ssam_filtered')
print(' --- filter_celltypemaps done, vf shape: ', analysis.dataset.vf.shape)
ds.plot_celltypes_map(rotate=1, set_alpha=False) # SSAM plotting function
plt.savefig(f'ssam_celltype_map_filtered_um_per_pixel={um_per_pixel}_bd={bd}.png', dpi = 300)

df.to_csv(f'ssam_celltype_maps_um_per_pixel={um_per_pixel}_bd={bd}.tsv', sep = '\t')