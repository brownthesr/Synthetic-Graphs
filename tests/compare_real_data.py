"""This file is for comparing how well the predictions for the real world datasets
did side by side
"""

from telnetlib import XASCII
from tkinter.messagebox import YES
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.spatial import Delaunay
import numpy as np
from scipy.ndimage import convolve


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

num_classess = [2,3]
DC = False
model = "GCN"
show_positive = True
show_negative = False
fig = plt.figure()
gs = GridSpec(10,31)
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[0:10,0:10])
ax_main[1] = fig.add_subplot(gs[0:10,10:20])#,sharey = ax_main[0])
ax_main[2] = fig.add_subplot(gs[0:10,20:30])#,sharey = ax_main[0])
# ax_NN = fig.add_subplot(gs[:10,30],sharey = ax_main[0])
# ax_Spectral = [1,2,3,4]
# ax_Spectral[0] = fig.add_subplot(gs[10,0:10],sharex=ax_main[0])
# ax_Spectral[1] = fig.add_subplot(gs[10,10:20],sharex=ax_main[0])
# ax_Spectral[2] = fig.add_subplot(gs[10,20:30],sharex=ax_main[0])
fig.subplots_adjust(wspace = 0,hspace=0)
# ax_main[].set_yticks([])
# ax_main[2].set_yticks([])
# ax_Spectral[0].set_yticks([])
# ax_Spectral[1].set_yticks([])
# ax_Spectral[2].set_yticks([])
p = 0
datasets = ["Cora","DBLP","Photo"]
mapper = {"Cora":7,"DBLP":4,"Photo":8}
DC = True
for data_name in datasets:
    num_classes = mapper[data_name]
    f2 = f"data/averaged_runs/Masters/{3}_NN.txt"
    if DC:
        f1 = f"data/averaged_runs/Masters/transductive_{data_name}_{num_classes}_DC_{model}.txt"
        f3 = f"data/averaged_runs/Masters/{3}_DC_Spectral.txt"
    else:
        f1 = f"data/averaged_runs/Masters/{num_classes}_{model}.txt"
        f3 = f"data/averaged_runs/Masters/{num_classes}_Spectral.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_vanilla = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)
    #print(len(test_accs))
    print(test_accs.shape)
    #note put the NN and the eigenvector in test_accs_vanilla
    x = test_accs[:,1]# lambda
    z = test_accs[:,0] #accs
    row_size = 121
    z = z.reshape(200,row_size)
    x = x.reshape(200,row_size)
    y = test_accs[:,2]# mu
    y = y.reshape(200,row_size)
    y = y
    first_z = z

    NN_x = test_accs_vanilla[:,2]
    NN_c = test_accs_vanilla[:,0]
    NN_y = test_accs_vanilla[:,1]
    S_x = test_accs_spectral[:,2]
    S_c = test_accs_spectral[:,0]
    S_y = test_accs_spectral[:,1]
    #y = y.reshape(200,61)
    print(test_accs_vanilla.shape)
    f1 = f1[27:-4]
    f2 = f2[27:-4]

    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    im_main = ax_main[p].scatter(x,y,c=z,cmap="coolwarm",vmin=0,vmax=1)

    plt.colorbar(im_main, ax=ax_main,cax = fig.add_subplot(gs[0:10,30]))
    ax_main[p].set_title(f"{data_name} accuracy")
    ax_main[p].grid(color="white")
    ax_main[0].set_ylabel("Feature information")
    ax_main[1].set_xlabel("Edge information ")
    if data_name == "Cora":
        ax_main[p].scatter([10.25],[0.37839325207321856],c=np.array([.813]),s=200,cmap="coolwarm",vmin=0,vmax=1,edgecolors="k")
        ax_main[p].grid(False)
    elif data_name == "DBLP":
        ax_main[p].scatter([.565],[0.118],c=np.array([.7]),s=200,cmap="coolwarm",vmin=0,vmax=1,edgecolors="k")
        ax_main[p].grid(False)
    elif data_name == "Photo":
        ax_main[p].scatter([10.53],[2.95],c=np.array([.8]),s=200,cmap="coolwarm",vmin=0,vmax=1,edgecolors="k")
        ax_main[p].grid(False)
    p += 1

plt.show()
