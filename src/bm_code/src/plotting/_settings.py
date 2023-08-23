import os
from typing import NamedTuple

class _PLOTTING_KEYS(NamedTuple):
    COLORS_METHODS: dict = {
        'Raw': '#969696',
        'Watershed': '#C7E9C0', 
        'Cellpose': '#529E3E',  
        'ClusterMap': '#AEC7E8', 
        'ClusterMap (no image)': '#AEC7E8', 
        'ClusterMap (w. image)': '#3B75AF', 
        'Baysor': '#FDD0A2',
        'Baysor (no prior)': '#FDD0A2', 
        'Baysor (w. prior)': '#FBAE6B', 
        'SSAM': '#99CCFF',
        'Tacco': '#FF99CC',
        'Bering': '#C43932'
    }
    XCOLS_DICT: dict = {
        'Raw': 'x',
        'Watershed': 'x',
        'Cellpose': 'x',
        'ClusterMap': 'x',
        'ClusterMap (no image)': 'spot_location_1',
        'ClusterMap (w. image)': 'spot_location_1',
        'Baysor': 'x',
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
        'Baysor': 'y',
        'Baysor (no prior)': 'y',
        'Baysor (w. prior)': 'y',
        'Bering': 'y'
    }
    PTS_SIZE_SLICE_BG = 0.005
    PTS_SIZE_SLICE_FG = 0.01
    PTS_SIZE_WINDOW_BG = 0.15
    PTS_SIZE_WINDOW_FG = 0.15
    MARKERSCALE: float = 30.0
    LEGENDFONTSIZE: float = 12.0
    TITLESIZE: float = 20.0
    CELLID_TEXTSIZE: float = 7.5

PLOT_KEYS = _PLOTTING_KEYS()