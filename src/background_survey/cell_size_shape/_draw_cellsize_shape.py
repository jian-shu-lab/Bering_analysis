import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

def get_convex(x, y):
    from scipy.spatial import ConvexHull
    points = np.vstack((x, y)).T
    hull = ConvexHull(points)
    area = hull.area
    axis_ratio = hull.volume / hull.area
    return points[hull.vertices], area, axis_ratio

def get_cellsize_shape(x, y, cellsize):
    from shapely.geometry import Polygon
    from shapely.ops import cascaded_union
    from shapely.geometry import Point
    from shapely.geometry import MultiPoint
    from shapely.geometry import MultiPolygon
    from shapely.geometry import LineString

    # get convex hull
    points, area, axis_ratio = get_convex(x, y)
    # get cellsize shape
    cellsize_shape = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i+1)%len(points)]
        # get line
        line = LineString([p1, p2])
        # get cellsize shape
        cellsize_shape.append(line.buffer(cellsize))
    # union cellsize shape
    cellsize_shape = cascaded_union(cellsize_shape)
    return cellsize_shape, area, axis_ratio

if __name__ == '__main__':
    # get data
    df = pd.read_csv('/data/aronow/Kang/spatial/Bering/benchmark/bm_data/bm2_lungtumor_cosmx_he_et_al/Lung5_Rep1_tx_fov_10.txt', sep='\t', header=0,index_col=0)
    labels = df['labels'].values
    n_cells_per_label = 3
    for label in np.unique(labels):
        df_label = df[df['labels']==label].copy()
        print(label, df_label.shape[0])
        cells = df_label['segmented'].unique()

        for i in range(n_cells_per_label):
            # get cell
            try:
                cell = cells[i]
            except:
                continue
            # get cell data
            df_label_cell = df_label[df_label['segmented']==cell].copy()
            # get x, y
            x = df_label_cell['x']
            y = df_label_cell['y']
            if len(x) > 500:
                continue
            # get cellsize shape
            cellsize_shape, area, axis_ratio = get_cellsize_shape(x, y, cellsize=0.1)
            # plot
            fig, ax = plt.subplots()
            ax.plot(x, y, 'o')
            ax.add_patch(mpl.patches.Polygon(np.array(cellsize_shape.exterior), alpha=0.5))
            ax.set_aspect('equal')
            plt.savefig('figures/cellsize_shape_{}_{}.png'.format(label, cell))