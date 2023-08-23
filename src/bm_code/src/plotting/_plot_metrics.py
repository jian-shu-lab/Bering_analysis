import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 20

from ._settings import PLOT_KEYS as K

def plot_mutual_information(methods, values, output_name = None):
    '''
    Plot mutual information between segmentation and spot detection
    '''
    # plot mutual information
    colors = [K.COLORS_METHODS[i] for i in methods]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Adjusted mutual information', fontsize=16)
    ax.set_xlabel('Segmentation method', fontsize=16)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    # ax.set_ylim(0, 1)
    if output_name is not None:
        fig.savefig(output_name, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Adjusted_mutual_information.pdf', bbox_inches = 'tight')
    return fig

def plot_adjusted_rand_index(methods, values, output_name = None):
    '''
    Plot adjusted rand index between segmentation and spot detection
    '''
    # plot mutual information
    colors = [K.COLORS_METHODS[i] for i in methods]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Adjusted Rand Index', fontsize=16)
    ax.set_xlabel('Segmentation method', fontsize=16)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    # ax.set_yticklabels(ax.get_yticks(), fontsize=14)
    # ax.set_ylim(0, 1)
    if output_name is not None:
        fig.savefig(output_name, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Adjusted_rand_index.pdf', bbox_inches = 'tight')
    return fig

def plot_accuracy_foreground(methods, values, output_name = None):
    '''
    Plot adjusted rand index between segmentation and spot detection
    '''
    # plot number segmented cells
    colors = [K.COLORS_METHODS[i] for i in methods]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy of fore/background prediction', fontsize=16)
    ax.set_xlabel('Segmentation method', fontsize=16)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    
    if output_name is not None:
        fig.savefig(output_name, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Accuracy_foreground_background_prediction.pdf', bbox_inches = 'tight')
    return fig


def plot_watershed(img, markers, output_name = None):
    '''
    Plot watershed segmentation
    '''
    # plot watershed segmentation
    fig, ax = plt.subplots(figsize=(20,15))
    ax.imshow(img, cmap='gray')
    ax.imshow(markers, cmap='jet', alpha=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if output_name is not None:
        fig.savefig(output_name, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Watershed_segmentation.png', bbox_inches = 'tight')
    return fig

def plot_num_cells(methods, values, output_name = None):
    '''
    Plot adjusted rand index between segmentation and spot detection
    '''
    # plot number segmented cells
    colors = [K.COLORS_METHODS[i] for i in methods]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylabel('Number of cells', fontsize=16)
    ax.set_xlabel('Segmentation method', fontsize=16)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    
    if output_name is not None:
        fig.savefig(output_name, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Number_of_cells.pdf', bbox_inches = 'tight')
    return fig

def plot_fraction_assigned_mols(methods, values, output_name = None):
    '''
    Plot fraction of assigned molecules
    '''
    # plot number segmented cells
    colors = [K.COLORS_METHODS[i] for i in methods]
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(methods, values, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fraction of assigned molecules', fontsize=16)
    ax.set_xlabel('Segmentation method', fontsize=16)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    
    if output_name is not None:
        fig.savefig(output_name, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Fraction_assigned_molecules.pdf', bbox_inches = 'tight')
    return fig

def plot_nodeclf_accuracy(methods, values_celltype, values_foreground, output_name_celltype = None, output_name_foreground = None):
    colors = [K.COLORS_METHODS[i] for i in methods]
    # cell type accuracy
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(methods, values_celltype, color=colors, edgecolor='black', linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy of cell type prediction', fontsize=16)
    ax.set_xlabel('Molecule annotation method', fontsize=16)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    if output_name_celltype is not None:
        fig.savefig(output_name_celltype, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Nodeclf_accuracy_celltype.pdf', bbox_inches = 'tight')
    # foreground accuracy
    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(methods, values_foreground, color=colors)
    ax.set_ylabel('Accuracy of fore/background prediction', fontsize=16)
    ax.set_xlabel('Molecule annotation method', fontsize=16)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    if output_name_foreground is not None:
        fig.savefig(output_name_foreground, bbox_inches = 'tight')
    else:
        fig.savefig('figures/Nodeclf_accuracy_foreground.pdf', bbox_inches = 'tight')