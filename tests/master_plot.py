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


COMPARE = False# this is for the comparison of the two methods
NN = False
EIG = False
f1 = "data/averaged_runs/Masters/7_GCN.txt"
f2 = "data/averaged_runs/Masters/7_NN.txt"
f3 = "data/averaged_runs/Masters/7_Spectral.txt"
test_accs = np.genfromtxt(f1)
test_accs_vanilla = np.genfromtxt(f2)
test_accs_spectral = np.genfromtxt(f3)
#print(len(test_accs))
print(test_accs.shape)
#note put the NN and the eigenvector in test_accs_vanilla
x = test_accs[:,1]# lambda
z = test_accs[:,0] #accs
z = z.reshape(200,121)
x = x.reshape(200,121)
y = test_accs[:,2]# mu
y = y.reshape(200,121)
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
fig = plt.figure()
gs = GridSpec(11,12)
ax_main = fig.add_subplot(gs[0:10,0:10])
ax_NN = fig.add_subplot(gs[:10,10],sharey = ax_main)
ax_Spectral = fig.add_subplot(gs[10,0:10],sharex=ax_main)
fig.subplots_adjust(wspace = 0,hspace=0)
ax_NN.set_xticks([])
#ax_NN.set_yticks([])
ax_Spectral.set_yticks([])
kernel = np.array([[1,4,7,4,1],
                   [4,16,26,16,4],
                   [7,26,41,26,7],
                   [4,16,26,16,4],
                   [1,4,7,4,1]])
z = convolve(z,kernel)/kernel.sum()

new_Z = np.zeros((200,121))
for i in range(200):
    new_Z[i] = z[i] - test_accs_vanilla[i][0]
compare_NN= new_Z



compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)
compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)

improvement_threshold = .01

idx = compare_NN < -improvement_threshold
xpoints=x[idx]
ypoints=y[idx]
points = np.vstack([xpoints,ypoints]).T
if len(points )> 4:
    edges = alpha_shape(points,0.05)
    for a,(i,j) in enumerate(edges):
        ax_main.plot(points[[i, j], 0], points[[i, j], 1],"w",linewidth=2)
idx = compare_NN > improvement_threshold
xpoints=x[idx]
ypoints=y[idx]
points = np.vstack([xpoints,ypoints]).T
if len(points )> 4:
    edges = alpha_shape(points,0.05)
    for a,(i,j) in enumerate(edges):
        ax_main.plot(points[[i, j], 0], points[[i, j], 1],"w",linewidth=2)

new_x = x[:,60:]
new_y = y[:,60:]
new_Z = np.zeros((61,200))
for i in range(61):
    new_Z[i] = z[:,60+i] - test_accs_spectral[i][0]
    #print(test_accs_vanilla[i][0])
compare_Spectral = new_Z.T
compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
idx = compare_Spectral < -improvement_threshold
xpoints=new_x[idx]
ypoints=new_y[idx]
points = np.vstack([xpoints,ypoints]).T
edges = alpha_shape(points,0.05)
for a,(i,j) in enumerate(edges):
    ax_main.plot(points[[i, j], 0], points[[i, j], 1],"k",linewidth=2)
idx = compare_Spectral > improvement_threshold
xpoints=new_x[idx]
ypoints=new_y[idx]
points = np.vstack([xpoints,ypoints]).T
edges = alpha_shape(points,0.05)
for a,(i,j) in enumerate(edges):
    ax_main.plot(points[[i, j], 0], points[[i, j], 1],"k",linewidth=2)

# this makes it look prettier

im_main = ax_main.scatter(x,y,c=z,cmap="coolwarm",vmin=.5,vmax=1)
ax_NN.scatter(NN_y,NN_x,c=NN_c,cmap="coolwarm",vmin=0.5,vmax=1)
ax_Spectral.scatter(S_y,S_x,c=S_c,cmap="coolwarm",vmin=0.5,vmax=1)
ax_Spectral.plot([-3,0],[0,0],"--k")
plt.colorbar(im_main, ax=ax_main,cax = fig.add_subplot(gs[0:10,11]))

ax_main.set_title(f"{f1} accuracy")
ax_main.grid(color="white")
ax_main.set_ylabel("Feature cloud distance from origin")
ax_NN.set_title(f"{f2} accuracy")
ax_Spectral.set_title("Spectral clustering accuracy")
ax_Spectral.set_xlabel("Normalized degree separation")
plt.show()
