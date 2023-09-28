"""This file is specifically made to generate a graph that compares all 4 of the models
side by side. It then compares degree-corrected vs poisson
"""

from telnetlib import XASCII
from tkinter.messagebox import YES
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.spatial import Delaunay
import numpy as np
from scipy.ndimage import convolve
import seaborn as sns

# We are plotting non-degree corrected data
num_classes = 2
DC = False
models = ["GCN","GAT","SAGE","Transformer"]
sns.set()
palette = sns.color_palette("colorblind")

# Setting up the figure
fig = plt.figure()
gs = GridSpec(22,43)
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[0:10,0:10])
ax_main[1] = fig.add_subplot(gs[0:10,10:20])
ax_main[2] = fig.add_subplot(gs[0:10,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[0:10,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)

# Looping through all the models
for p,model in enumerate(models):
    # Loading the data
    f2 = f"data/maxed_runs/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/maxed_runs/{num_classes}_DC_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_DC_Spectral.txt"
    else:
        f1 = f"data/maxed_runs/{num_classes}_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_Spectral.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)

    # obtaining the data
    z = test_accs[:,0] #accs
    x = test_accs[:,1]# lambda
    y = test_accs[:,2]# mu

    # Reshaping the data
    row_size = 121
    z = z.reshape(200,row_size)
    x = x.reshape(200,row_size)
    y = y.reshape(200,row_size)

    NN_x = test_accs_nn[:,2]
    NN_c = test_accs_nn[:,0]
    NN_y = test_accs_nn[:,1]
    S_x = test_accs_spectral[:,2]
    S_c = test_accs_spectral[:,0]
    S_y = test_accs_spectral[:,1]

    # Smoothing the data with a gaussian filter
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # Setting up titles
    im_main = ax_main[p].scatter(x,y,c=z,cmap="coolwarm",vmin=1/2,vmax=1)
    ax_main[p].set_title(f"{num_classes} Class {model}")
    ax_main[p].set_ylim(0,1)
    ax_main[0].set_xticks([])
    ax_main[0].set_ylabel("Non-Scale-Free\nFeature Information")

# Setting up x and y ticks
ax_main[0].set_yticks([0,.5,1])
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])
ax_main[0].set_xticks([])
ax_main[1].set_xticks([])
ax_main[2].set_xticks([])
ax_main[3].set_xticks([])

# Now we are plotting the degree corrected data
# Sets up various hyperparameters
num_classes = 2
DC = True
models = ["GCN","GAT","SAGE","Transformer"]
show_positive = False
show_negative = False

# Setting up the figure
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[11:21,0:10])
ax_main[1] = fig.add_subplot(gs[11:21,10:20])
ax_main[2] = fig.add_subplot(gs[11:21,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[11:21,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])

# Looping through all the models
for p,model in enumerate(models):
    # Loading the data
    f2 = f"data/maxed_runs/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/maxed_runs/{num_classes}_DC_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_DC_Spectral.txt"
    else:
        f1 = f"data/maxed_runs/{num_classes}_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_Spectral.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)
    
    z = test_accs[:,0] #accs
    x = test_accs[:,1]# lambda
    y = test_accs[:,2]# mu

    # Reshaping the Data
    row_size = 121
    z = z.reshape(200,row_size)
    x = x.reshape(200,row_size)
    y = y.reshape(200,row_size)

    NN_x = test_accs_nn[:,2]
    NN_c = test_accs_nn[:,0]
    NN_y = test_accs_nn[:,1]
    S_x = test_accs_spectral[:,2]
    S_c = test_accs_spectral[:,0]
    S_y = test_accs_spectral[:,1]
    
    # Smoothing the data with a gaussian filter
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # Setting up titles
    im_main = ax_main[p].scatter(x,y,c=z,cmap="coolwarm",vmin=1/2,vmax=1)
    colorbars = fig.add_subplot(gs[5:15,42])
    clb = plt.colorbar(im_main, ax=ax_main,cax = colorbars)
    # clb.ax.set_title("Colorbar")
    ax_main[p].set_xlabel(f"Edge Information")
    ax_main[p].set_ylim(0,1)
    ax_main[0].set_ylabel("Scale Free\nFeature Information")

# Setting the x-ticks
ax_main[0].set_yticks([0,.5,1])

plt.show()
