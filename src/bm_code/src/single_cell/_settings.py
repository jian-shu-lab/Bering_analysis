from typing import NamedTuple

class _SC_KEYS(NamedTuple):
    GENE_COLNAME_DICT: dict = {
        'Raw': 'features',
        'Baysor': 'gene',
        'Baysor (no prior)': 'gene',
        'Baysor (w. prior)': 'gene',
        'Bering': 'features',
        'ClusterMap': 'gene_name',
        'ClusterMap (no image)': 'gene_name',
        'ClusterMap (w. image)': 'gene_name',
        'Cellpose': 'features',
        'Watershed': 'features',
    }
    CELLID_COLNAME_DICT: dict = {
        'Raw': 'segmented',
        'Baysor': 'cell',
        'Baysor (no prior)': 'cell',
        'Baysor (w. prior)': 'cell',
        'Bering': 'ensembled_cells',
        'ClusterMap': 'clustermap',
        'ClusterMap (no image)': 'clustermap',
        'ClusterMap (w. image)': 'clustermap',
        'Cellpose': None,
        'Watershed': None,
    }
    XCOLS_DICT: dict = {
        'Raw': 'x',
        'Watershed': 'x',
        'Cellpose': 'x',
        'ClusterMap': 'x',
        'ClusterMap (no image)': 'spot_location_1',
        'ClusterMap (w. image)': 'spot_location_1',
        'Baysor (no prior)': 'x',
        'Baysor (w. prior)': 'x',
        'Bering': 'x'
    }
    YCOLS_DICT: dict = {
        'Raw': 'y',
        'Watershed': 'y',
        'Cellpose': 'y',
        'ClusterMap': 'y',
        'ClusterMap (no image)': 'spot_location_2',
        'ClusterMap (w. image)': 'spot_location_2',
        'Baysor (no prior)': 'y',
        'Baysor (w. prior)': 'y',
        'Bering': 'y'
    }

SC_KEYS = _SC_KEYS()