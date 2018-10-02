import numpy as np
import pandas as pd
from pyntcloud import PyntCloud as pc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Colors import getDistinctColors


def plot(pointclouds, width=900, initial_point_size=0.01):
    if len(pointclouds) > 1:
        s = None
        for ps in pointclouds:
            p = pd.DataFrame(ps.astype(np.float32), columns=['x', 'y', 'z', 'red', 'green', 'blue'])
            c = pc(p)
            if s is None:
                s = c.plot(width=width, initial_point_size=initial_point_size, return_scene=True)
            else:
                s = c.plot(width=width, initial_point_size=initial_point_size, return_scene=True, scene=s)
        

    else:
        p = pd.DataFrame(pointclouds[0].astype(np.float32), columns=['x', 'y', 'z', 'red', 'green', 'blue'])
        c = pc(p)
        c.plot(width=width, initial_point_size=initial_point_size)

def plot_points(ps, size=(15, 8)):
    plot_ps = None
    colors = None
    possible_colors = getDistinctColors(len(ps))
    colors = []
    for c, p in enumerate(ps):
        if plot_ps is None:
            plot_ps = p
        else:
            plot_ps = np.concatenate((plot_ps, p))
        for _ in range(len(p)):
            colors.append(c)
    
    print(colors)
    xs = plot_ps[:,0]
    ys = plot_ps[:,1]
    zs = plot_ps[:,2]
    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=colors, marker='x')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()