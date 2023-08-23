import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 20

from ._settings import PLOT_KEYS as K

def _get_window_centroids(df_spots, number_windows = 9, window_size = 1000, min_spots = 100):
    '''
    Define windows for plotting spots
    '''
    # define windows
    if type(number_windows) is not int:
        raise ValueError('number_windows must be an integer')

    num_window_axis = int(np.sqrt(number_windows))

    x_min, x_max = df_spots.x.min() + window_size / 2, df_spots.x.max() - window_size / 2
    y_min, y_max = df_spots.y.min() + window_size / 2, df_spots.y.max() - window_size / 2

    x_range = x_max - x_min
    y_range = y_max - y_min

    x_step = x_range / num_window_axis - 0.01
    y_step = y_range / num_window_axis - 0.01

    x_centroids = np.arange(x_min, x_max, x_step)
    y_centroids = np.arange(y_min, y_max, y_step)

    centroids = np.array(np.meshgrid(x_centroids, y_centroids)).T.reshape(-1,2)

    centroids_filtered = []
    for centroid in centroids:
        cx, cy = centroid
        df_spots_in_window = df_spots[(df_spots.x > cx - window_size / 2) & (df_spots.x < cx + window_size / 2) & (df_spots.y > cy - window_size / 2) & (df_spots.y < cy + window_size / 2)]
        if len(df_spots_in_window) >= min_spots:
            centroids_filtered.append(centroid)

    return centroids_filtered

def plot_slice_raw(ax, df_spots, color_by = 'foreground', window_size = None, window_centroid = None, bg_alpha = 0.2):
    if window_size is not None:
        pts_size_fg, pts_size_bg = K.PTS_SIZE_WINDOW_FG, K.PTS_SIZE_WINDOW_BG
        if window_centroid is None:
            raise ValueError('window_centroid must be specified if window_size is specified')
        else:
            x_ct, y_ct = window_centroid
            x_min, x_max = x_ct - window_size / 2, x_ct + window_size / 2
            y_min, y_max = y_ct - window_size / 2, y_ct + window_size / 2
            df_spots = df_spots[(df_spots.x > x_min) & (df_spots.x < x_max) & (df_spots.y > y_min) & (df_spots.y < y_max)].copy()
    else:
        pts_size_fg, pts_size_bg = K.PTS_SIZE_SLICE_FG, K.PTS_SIZE_SLICE_BG
        
    x, y = df_spots.x.values, df_spots.y.values
    cells = df_spots.segmented.values
    cell_types = df_spots.labels.values
    cells[cell_types == 'background'] = -1
    
    if color_by == 'foreground':
        labels = df_spots.labels.values
        color_dict = {k:np.random.rand(3,) for k in np.unique(labels) if k != 'background'}
        x_bg, y_bg = x[labels == 'background'], y[labels == 'background']
        x_fg, y_fg = x[labels != 'background'], y[labels != 'background']
        ax.scatter(x_bg, y_bg, c='#DCDCDC', s = pts_size_bg, label='background', alpha = bg_alpha)
        ax.scatter(x_fg, y_fg, c='#1F77B4', s = pts_size_fg, label='foreground')
        
        # color_dict = {k:np.random.rand(3,) for k in np.unique(cell_types) if k != 'background'}
        # color_dict['background'] = '#DCDCDC'

        # for cell_type in color_dict.keys():
        #     if cell_type != 'background':
        #         indices = np.where(cell_types == cell_type)[0]
        #         xc, yc = x[indices], y[indices]
        #         ax.scatter(xc, yc, c=color_dict[cell_type], s = pts_size_fg, label=cell_type)
        #     else:
        #         ax.scatter(x, y, c=color_dict[cell_type], s = pts_size_bg, label=cell_type, alpha = bg_alpha)

    elif color_by == 'cell':
        import pickle
        with open('cells.pl', 'wb') as f:
            pickle.dump(cells, f)
        color_dict = {k:np.random.rand(3,) for k in np.unique(cells)}
        for cell in color_dict.keys():
            indices = np.where(cells == cell)[0]
            xc, yc = x[indices], y[indices]
            if cell == -1: # background
                ax.scatter(xc, yc, c='#DCDCDC', s = pts_size_bg, label=cell, alpha = bg_alpha)
            else:
                ax.scatter(xc, yc, c=color_dict[cell], s = pts_size_fg, label=cell)
                ax.text(xc.mean(), yc.mean(), str(int(cell)), fontsize = K.CELLID_TEXTSIZE, c = 'black', ha = 'center', va = 'center')

    return ax, color_dict

def plot_slice_segmented(
    ax, df_results, mask = None, method = 'Bering', color_by = 'foreground',
    window_size = None, window_centroid = None,
    bg_alpha = 0.2
):
    print(method)
    color_dict = {'foreground': '#1F77B4', 'background': '#DCDCDC'}
    if mask is not None:
        mask = mask.copy().astype(np.int32)
    if window_size is not None:
        pts_size_fg, pts_size_bg = K.PTS_SIZE_WINDOW_FG, K.PTS_SIZE_WINDOW_BG
        if window_centroid is None:
            raise ValueError('window_centroid must be specified if window_size is specified')
        else:
            x_ct, y_ct = window_centroid
            x_min, x_max = x_ct - window_size / 2, x_ct + window_size / 2
            y_min, y_max = y_ct - window_size / 2, y_ct + window_size / 2
            if method not in ['ClusterMap', 'ClusterMap (no image)', 'ClusterMap (w. image)']:
                df_results = df_results[(df_results.x > x_min) & (df_results.x < x_max) & (df_results.y > y_min) & (df_results.y < y_max)].copy()
            else:
                df_results = df_results[(df_results.spot_location_1 > x_min) & (df_results.spot_location_1 < x_max) & (df_results.spot_location_2 > y_min) & (df_results.spot_location_2 < y_max)].copy()
    else:
        pts_size_fg, pts_size_bg = K.PTS_SIZE_SLICE_FG, K.PTS_SIZE_SLICE_BG

    if method not in ['ClusterMap', 'ClusterMap (no image)', 'ClusterMap (w. image)']:
        x, y = df_results.x.values, df_results.y.values
    else:
        x, y = df_results.spot_location_1.values, df_results.spot_location_2.values
    
    if method == 'Bering':
        cell_types_pred = df_results.predicted_node_labels.values
        cell_types_predEnsem = df_results.ensembled_labels.values
        if color_by == 'foreground':
            values = np.where(cell_types_predEnsem != 'Unknown', 'foreground', 'background')
        elif color_by == 'cell':
            values = df_results.ensembled_cells.values.copy()
    elif method == 'Baysor' or method == 'Baysor (no prior)' or method == 'Baysor (w. prior)':
        if color_by == 'foreground':
            values = np.where(df_results['cell'].values == 0, 'background', 'foreground')
        elif color_by == 'cell':
            values = df_results.cell.values.copy()
            values[values == 0] = -1 # make background -1
    elif method == 'ClusterMap' or method == 'ClusterMap (no image)' or method == 'ClusterMap (w. image)':
        if color_by == 'foreground':
            values = np.where(df_results['clustermap'].values == -1, 'background', 'foreground')
        elif color_by == 'cell':
            values = df_results.clustermap.values.copy()
    elif method in ['Cellpose', 'Watershed']:
        x_appro = x.astype(np.int16)
        y_appro = y.astype(np.int16)
        if mask is None:
            raise ValueError('Mask is required for Cellpose and Watershed')
        if color_by == 'foreground':
            values = np.where(mask[y_appro, x_appro] == 0, 'background', 'foreground')
        elif color_by == 'cell':
            values = mask[y_appro, x_appro].copy()
            values[values == 0] = -1 # make background -1

    if color_by == 'foreground':
        for seg_group in color_dict.keys():
            indices = np.where(values == seg_group)[0]
            xc, yc = x[indices], y[indices]
            if seg_group == 'background':
                ax.scatter(xc, yc, c=color_dict[seg_group], label=seg_group, alpha = bg_alpha, s = pts_size_bg)
            else:
                ax.scatter(xc, yc, c=color_dict[seg_group], label=seg_group, s = pts_size_fg)
    elif color_by == 'cell':
        if method == 'Cellpose':
            import pickle
            with open('cellpose_mask.pkl', 'wb') as f:
                pickle.dump(mask, f)
            with open('cellpose_values.pkl', 'wb') as f:
                pickle.dump(values, f)
        for cell in np.unique(values):
            indices = np.where(values == cell)[0]
            xc, yc = x[indices], y[indices]

            if cell == -1:
                ax.scatter(xc, yc, c=color_dict['background'], label='background', alpha = bg_alpha, s = pts_size_bg)
            else:
                ax.scatter(xc, yc, c=np.random.rand(3,), label='cell {}'.format(cell), s = pts_size_fg)
                ax.text(xc.mean(), yc.mean(), str(int(cell)), fontsize = K.CELLID_TEXTSIZE, c = 'black', ha = 'center', va = 'center')

    return ax, color_dict

def plot_slice(
    df_spots, df_results_list, masks_list, method_list, color_by = 'foreground',
    window_size = None, num_windows = 9, window_centroid_x = None, window_centroid_y = None,
    bg_alpha = 0.2, axis_size = 4, num_rows = 2, save = False, save_name = None
):
    if window_size is not None:
        window_centroids = _get_window_centroids(df_spots, number_windows = num_windows, window_size = window_size)
    else:
        window_centroids = [None]

    for window_centroid in window_centroids:
        if window_centroid is not None:
            window_centroid_x, window_centroid_y = window_centroid

        num_subplots = len(df_results_list) + 1
        
        num_rows = num_rows
        num_cols = 4

        if num_rows == 2:
            fig, axes = plt.subplots(num_rows, num_cols, figsize = (axis_size * num_cols, axis_size * num_rows), sharex = True, sharey = True)
            axes[0,0], color_dict = plot_slice_raw(axes[0,0], df_spots, color_by = color_by, window_size = window_size, window_centroid = (window_centroid_x, window_centroid_y), bg_alpha = bg_alpha)
            axes[0,0].set_title('Raw', fontsize = K.TITLESIZE)
            # if color_by == 'foreground':
            #     axes[0,0].legend(markerscale = K.MARKERSCALE, fontsize = K.LEGENDFONTSIZE, loc = 'upper right')
            for i, (df_results, mask, method) in enumerate(zip(df_results_list, masks_list, method_list)):
                axes[(i+1)//num_cols, (i+1)%num_cols], _ = plot_slice_segmented(axes[(i+1)//num_cols, (i+1)%num_cols], df_results, mask, method, color_by = color_by, window_size = window_size, window_centroid = (window_centroid_x, window_centroid_y), bg_alpha = bg_alpha)
                axes[(i+1)//num_cols, (i+1)%num_cols].set_title(method, fontsize = K.TITLESIZE)
                # if color_by == 'foreground':
                #     axes[(i+1)//num_cols, (i+1)%num_cols].legend(markerscale = K.MARKERSCALE, fontsize = K.LEGENDFONTSIZE)
                #     # axes[(i+1)//num_cols, (i+1)%num_cols].set_title(method, fontsize = K.TITLESIZE)

            # for i in range(num_subplots, num_rows * num_cols):
            for i in range(num_rows * num_cols):
                # axes[i//num_cols, i%num_cols].axis('off')
                axes[i//num_cols, i%num_cols].set_xticks([])
                axes[i//num_cols, i%num_cols].set_yticks([])

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.10, wspace=0.10)
        
        elif num_rows == 1:
            fig, axes = plt.subplots(num_rows, num_cols, figsize = (axis_size * num_cols, axis_size * num_rows), sharex = True, sharey = True)
            axes[0], color_dict = plot_slice_raw(axes[0], df_spots, color_by = color_by, window_size = window_size, window_centroid = (window_centroid_x, window_centroid_y), bg_alpha = bg_alpha)
            axes[0].set_title('Raw', fontsize = K.TITLESIZE)
            # if color_by == 'foreground':
            #     axes[0].legend(markerscale = K.MARKERSCALE, fontsize = K.LEGENDFONTSIZE, loc = 'upper right')
            for i, (df_results, mask, method) in enumerate(zip(df_results_list, masks_list, method_list)):
                axes[i+1], _ = plot_slice_segmented(axes[i+1], df_results, mask, method, color_by = color_by, window_size = window_size, window_centroid = (window_centroid_x, window_centroid_y), bg_alpha = bg_alpha)
                axes[i+1].set_title(method, fontsize = K.TITLESIZE)

            for i in range(num_rows * num_cols):
                axes[i].set_xticks([])
            axes[0].set_yticks([])

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.10, wspace=0.10)

        if save:
            if save_name is None:
                if window_size is None:
                    fig.savefig('figures/Foreground_prediction_slice_comparison.png', bbox_inches = 'tight')
                else:
                    if color_by == 'foreground':
                        fig.savefig(f'figures/Foreground_prediction_window_comparison_size_{int(window_size)}_window_centroid_{int(window_centroid_x)}_{int(window_centroid_y)}.png', bbox_inches = 'tight')
                    elif color_by == 'cell':
                        fig.savefig(f'figures/Cell_prediction_window_comparison_size_{int(window_size)}_window_centroid_{int(window_centroid_x)}_{int(window_centroid_y)}.png', bbox_inches = 'tight')
            else:
                fig.savefig(save_name, bbox_inches = 'tight')