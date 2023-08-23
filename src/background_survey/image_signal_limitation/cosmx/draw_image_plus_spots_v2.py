import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def load_data(fov = 10):
    # load data
    df_spots_all = pd.read_csv(f'/data/aronow/Kang/spatial/Bering/cosmx/data/transcripts_aligned_celltypes/Lung5_Rep1_tx_fov_{fov}.txt', sep = '\t', header = 0, index_col = 0)

    fov_fold = 'F00' + str(fov) if fov < 10 else 'F0' + str(fov)
    img_dapi = tiff.imread(f'/data/aronow/Kang/spatial/Bering/cosmx/data/image/{fov_fold}/Lung5_Rep1_DAPI.tif')
    return df_spots_all, img_dapi

def draw_spots_strong_weak_intensity(
    df_spots_all, img_dapi, windows, min_intensities = 25, 
    figsize_main = (30,20), figsize_subplots = (10, 10),
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
    fig.savefig(f'lungtumor_dapi_spots_min={min_intensities}_mainFigure.png', dpi = 300, bbox_inches = 'tight')

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
        ax.scatter(x_window[signals_window < min_intensities], y_window[signals_window < min_intensities], s = 0.2, c = 'r')
        ax.scatter(x_window[signals_window >= min_intensities], y_window[signals_window >= min_intensities], s = 0.2, c = 'blue')
        
        ax.imshow(img_dapi_window, cmap = 'gray')
        # ax.invert_yaxis()
        # ax.set_facecolor('white')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        fig.savefig(f'lungtumor_dapi_spots_min={min_intensities}_Window{window_idx + 1}.png', dpi = 300, bbox_inches = 'tight')

if __name__ == '__main__':
    df_spots_all, img_dapi = load_data()
    tiff.imwrite('lungtumor_dapi.tif', img_dapi.astype(np.uint8))

    # x_rc, y_rc = 1300, 1900
    # width, height = 600, 600

    window1 = [1300, 1900, 600, 600]
    window2 = [1600, 200, 600, 600]
    window3 = [3000, 1700, 600, 600]
    window4 = [4700, 2000, 600, 600]

    windows = [window1, window2, window3, window4]
    draw_spots_strong_weak_intensity(df_spots_all, img_dapi, windows, min_intensities = 25)
