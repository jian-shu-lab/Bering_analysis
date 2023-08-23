import logging
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from cellpose import models, core, plot
from cellpose.io import logger_setup
logger = logging.getLogger(__name__)

def run_cellpose(img, channel = [0,0], model = 'cyto', diameter = None, flow_threshold = None, plot_name = None):
    mpl.rcParams['figure.dpi'] = 300
    use_GPU = core.use_gpu()
    
    # DEFINE CELLPOSE MODEL
    model = models.Cellpose(gpu=use_GPU, model_type=model)
    masks, flows, styles, diams = model.eval(img, diameter=diameter, flow_threshold=flow_threshold, channels=channel)

    nimg = 1

    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img, masks, flows[0], channels=channel)
    plt.tight_layout()
    if plot_name is None:
        fig.savefig('figures/Cellpose_segmentation.png', bbox_inches = 'tight')
    else:
        fig.savefig(plot_name, bbox_inches = 'tight')

    return masks, fig