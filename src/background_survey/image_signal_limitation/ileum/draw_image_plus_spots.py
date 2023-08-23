import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

def load_data(min_spots_cell = 20):
    img_dapi = tiff.imread('/data/aronow/Kang/spatial/data/MERFISH/Petukhov_et_al_2022_NatBiotechnology/data_release_baysor_merfish_gut/raw_data/dapi_stack.tif')
    img_dapi = np.max(img_dapi, axis = 0)

    df_spots_seg = pd.read_csv('/data/aronow/Kang/spatial/Bering/ileum/data/merfish_illeum_segmented_v2.tsv', sep = '\t', header = 0, index_col = 0)
    df_spots_seg.segmented = df_spots_seg.segmented.astype(str)

    # small cells
    cell_stats = df_spots_seg.groupby(['segmented']).size().sort_values()
    small_cells = (cell_stats[cell_stats < min_spots_cell]).index.values
    df_spots_seg = df_spots_seg.loc[~df_spots_seg['segmented'].isin(small_cells),:].copy()
    df_spots_seg_small = df_spots_seg.loc[df_spots_seg['segmented'].isin(small_cells),['x','y','z','features']].copy()

    df_spots_unseg = pd.read_csv('/data/aronow/Kang/spatial/Bering/ileum/data/merfish_illeum_unsegmented_v2.tsv', sep = '\t', header = 0, index_col = 0)
    df_spots_unseg = pd.concat([df_spots_unseg, df_spots_seg_small], axis = 0)
    df_spots_all = pd.concat([df_spots_seg, df_spots_unseg], axis = 0)
    # remove rows with labels as numpy nan
    df_spots_all = df_spots_all.loc[~pd.isna(df_spots_all.labels),:].copy()

    return df_spots_all, img_dapi

def draw_spots_strong_weak_intensity(
    df_spots_all, img_dapi, windows, min_intensities = 25, 
    figsize_main = (20,30), figsize_subplots = (10, 10),
):
    # draw spots with strong and weak intensity in the back
    x, y = df_spots_all.x.values, df_spots_all.y.values
    
    x_pixel, y_pixel = x.astype(int), y.astype(int)
    signals = img_dapi[y_pixel, x_pixel]

    # main figure (red - out of staining, blue - in staining)
    fig, ax = plt.subplots(1, 1, figsize = figsize_main)
    ax.scatter(x[signals < min_intensities], y[signals < min_intensities], s = 0.05, c = 'r')
    ax.scatter(x[signals >= min_intensities], y[signals >= min_intensities], s = 0.05, c = 'blue')
    ax.set_xlim(0, img_dapi.shape[1])
    ax.set_ylim(0, img_dapi.shape[0])
    # remove ticks and ticklabels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for window_idx, window in enumerate(windows):
        x_rc, y_rc = window[0], window[1]
        width, height = window[2], window[3]
        rect = Rectangle((x_rc, y_rc), width, height, linewidth = 4, edgecolor = 'black', facecolor = 'none')
        ax.add_patch(rect)
        # bold font text
        # ax.text(x_rc + width / 2, y_rc + height / 2, f'W{window_idx + 1}', fontsize = 50, color = 'black', ha = 'center', va = 'center', fontweight = 'bold')
        # ax.text(x_rc + 150, y_rc - 150, f'W{window_idx + 1}', fontsize = 50, color = 'black', ha = 'center', va = 'center', fontweight = 'bold')
    fig.savefig(f'merfish_ileum_dapi_spots_min={min_intensities}_mainFigure.png', dpi = 300, bbox_inches = 'tight')

    # subplots (zoom in)
    for window_idx, window in enumerate(windows):
        x_rc, y_rc = window[0], window[1]
        width, height = window[2], window[3]

        x_window = x[(x >= x_rc) & (x < x_rc + width - 0.5) & (y >= y_rc) & (y < y_rc + height - 0.5)]
        y_window = y[(x >= x_rc) & (x < x_rc + width - 0.5) & (y >= y_rc) & (y < y_rc + height - 0.5)]
        x_window = x_window - x_rc
        y_window = y_window - y_rc
        x_window_pixel, y_window_pixel = np.floor(x_window).astype(int), np.floor(y_window).astype(int)
        
        img_dapi_window = img_dapi[y_rc:y_rc + height, x_rc:x_rc + width]
        signals_window = img_dapi_window[y_window_pixel, x_window_pixel]

        # convert y coordinate to match the image
        y_window = height - y_window
        img_dapi_window = np.flipud(img_dapi_window)

        fig, ax = plt.subplots(1, 1, figsize = figsize_subplots)
        ax.scatter(x_window[signals_window < min_intensities], y_window[signals_window < min_intensities], s = 0.5, c = 'r')
        ax.scatter(x_window[signals_window >= min_intensities], y_window[signals_window >= min_intensities], s = 0.5, c = 'blue')
        
        ax.imshow(img_dapi_window, cmap = 'gray')
        # ax.invert_yaxis()
        # ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        fig.savefig(f'merfish_ileum_dapi_spots_min={min_intensities}_Window{window_idx + 1}.png', dpi = 300, bbox_inches = 'tight')


if __name__ == '__main__':
    df_spots_all, img_dapi = load_data()
    # draw overview slice and windows
    '''
    tiff.imwrite('merfish_illeum_dapi.tif', img_dapi.astype(np.uint8))

    x, y = df_spots_all.x.values, df_spots_all.y.values
    
    window1 = [1200, 4000, 600, 600]
    window2 = [1900, 200, 600, 600]
    window3 = [900, 6000, 600, 600]
    window4 = [3600, 7000, 600, 600]

    windows = [window1, window2, window3, window4]
    # windows = [window1, window2]
    draw_spots_strong_weak_intensity(df_spots_all, img_dapi, windows, min_intensities = 25)
    '''
    
    # show low signal spot ratios across cell types
    print(df_spots_all.head())
    print(df_spots_all.columns)
    print(img_dapi)

    x, y = df_spots_all.x.values.astype(np.int16), df_spots_all.y.values.astype(np.int16)
    df_spots_all['signal'] = img_dapi[y, x]
    df_spots_all['signal_type'] = np.where(df_spots_all['signal'] < 25, 'weak', 'strong')

    labels = []
    proportion_low_signal = []
    for label in np.unique(df_spots_all['labels']):
        df_label = df_spots_all[df_spots_all['labels'] == label]
        labels.append(label)
        proportion_low_signal.append(np.sum(df_label['signal_type'] == 'weak') / df_label.shape[0])

    # draw bar plots showing the proportion of low signal spots across cell types
    fig, ax = plt.subplots(1, 1, figsize = (5, 2.5))
    ax.bar(labels, proportion_low_signal, edgecolor = 'black')
    ax.set_xticklabels(labels, rotation = 90)
    ax.set_ylabel('Proportion of low signal spots')
    # ax.set_ylim(0,1)
    fig.savefig('merfish_ileum_proportion_low_signal_spots.pdf', dpi = 300, bbox_inches = 'tight')