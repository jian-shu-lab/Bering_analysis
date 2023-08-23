import numpy as np
import pandas as pd

def _get_accuracy_nodeclass(df_results, method = 'Bering'):
    '''
    Accuracy of cell types in Bering / Baseline / SSAM / TACCO results
    '''
    if method == 'Bering':
        labels = df_results['labels'].values
        labels_pred = df_results['ensembled_labels'].values
    elif method == 'Tacco':
        labels = df_results['labels'].values
        labels_pred = df_results['tacco_labels'].values
    elif method == 'Baseline':
        1
    elif method == 'SSAM':
        1
    
    # get segmentation groups
    groups = np.where(labels == 'background', 'background', 'foreground')
    groups_pred = np.where(labels_pred == 'background', 'background', 'foreground')

    accu_group = np.sum(groups == groups_pred) / len(groups)
    foreground_indices = np.where(labels != 'background')[0] # only calculate accuracy for foreground
    accu_cell = np.sum(labels[foreground_indices] == labels_pred[foreground_indices]) / len(foreground_indices)

    return accu_group, accu_cell