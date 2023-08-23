from typing import NamedTuple

class _STATS_KEYS(NamedTuple):
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

STAT_KEYS = _STATS_KEYS()