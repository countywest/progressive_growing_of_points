import os
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='z',
                         xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):
    if sizes is None:
        sizes = [1.0 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30
        azim = -45 + 90 * i
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def plot_pcd_three_views_colorful(filename, pcds, titles, suptitle='', num_colors=[1,2,4,8],
                                             is_contiguous=True, sizes=None, zdir='z', elev=30, azims=[-45, 45, 135],
                                             xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3)):

    colors = ['red', 'dodgerblue', 'forestgreen', 'darkorchid', 'magenta', 'aqua', 'limegreen', 'm']

    def sort(color_list): # only for is_contiguous == True
        if len(color_list) == 2:
            return color_list
        else:
            mid = len(color_list) // 2
            left = sort(color_list[:mid])
            right = sort(color_list[mid:])
            ret = []
            for i in range(len(left)):
                ret.append(left[i])
                ret.append(right[i])
            return ret

    if sizes is None:
        sizes = [1.0 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        azim = azims[i]
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            num_points = pcd.shape[0]
            idx = np.array(range(num_points))
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)

            # input
            if num_colors[j] == 1:
                if zdir == 'y':
                    c = pcd[:, 2]
                elif zdir == 'z':
                    c = pcd[:, 1]
                ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
                           zdir=zdir, cmap='jet', c=c, s=size, vmin=-0.5, vmax=0.5)

            # outputs
            else:
                temp_colors = sort(colors[:num_colors[j]]) if is_contiguous else colors[:num_colors[j]]
                for k, color in enumerate(temp_colors):
                    if is_contiguous:
                        partial_pcd = pcd[np.logical_and(k * num_points // num_colors[j] <= idx,
                                                 idx < (k + 1) * num_points // num_colors[j])]
                    else:
                        partial_pcd = pcd[idx % len(temp_colors) == k]
                    ax.scatter(partial_pcd[:, 0], partial_pcd[:, 1], partial_pcd[:, 2],
                               zdir=zdir, c=np.array([color]), s=size, vmin=-0.5, vmax=0.5)

            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)


    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)

def plot_pcd(config, dir, filename, pcds, titles, suptitle='', num_colors=[1,2,4,8]):
    os.makedirs(dir, exist_ok=True)

    if config['dataset']['type'] == 'shapenet':
        plot_pcd_three_views_colorful(filename=os.path.join(dir, filename), pcds=pcds, titles=titles, suptitle=suptitle,
                                      num_colors=num_colors, sizes=None,
                                      zdir='z', xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3))
    elif config['dataset']['type'] == 'pcn':
        plot_pcd_three_views_colorful(filename=os.path.join(dir, filename), pcds=pcds, titles=titles, suptitle=suptitle,
                                      num_colors=num_colors, sizes=None,
                                      zdir='y', xlim=(-0.3, 0.3), ylim=(-0.3, 0.3), zlim=(-0.3, 0.3))
    elif config['dataset']['type'] == 'topnet':
        plot_pcd_three_views_colorful(filename=os.path.join(dir, filename), pcds=pcds, titles=titles, suptitle=suptitle,
                                  num_colors=num_colors, sizes=None,
                                  zdir='y', xlim=(-0.4, 0.4), ylim=(-0.4, 0.4), zlim=(-0.4, 0.4))
    else:
        raise NotImplementedError