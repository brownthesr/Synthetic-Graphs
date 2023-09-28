"""This file is specifically made to generate a graph that compares all 4 of the models
side by side.
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



# We group the plots into 2 sets of figures just for sizing constraints

# first we plot class sizes of 2 and 3

maxes = True
if maxes:
    folder = "maxed_runs"
else:
    folder = "averaged_runs/Masters"

# Set up the grid and parameters
gs = GridSpec(23,43)
fig = plt.figure()
num_class = [2,3]
DC = False
sns.set()
palette = sns.color_palette("colorblind")

for k,classes in enumerate(num_class):

    # Set up parameters
    models = ["GCN","GAT","SAGE","Transformer"]

    # Add subplots for GNN
    ax_main = [1,2,3,4]
    ax_main[0] = fig.add_subplot(gs[0+12*k:10+12*k,0:10])
    ax_main[1] = fig.add_subplot(gs[0+12*k:10+12*k,10:20])
    ax_main[2] = fig.add_subplot(gs[0+12*k:10+12*k,20:30],sharey = ax_main[1])
    ax_main[3] = fig.add_subplot(gs[0+12*k:10+12*k,30:40],sharey = ax_main[1])

    # Add subplots for NN and Spectral clustering
    ax_NN = fig.add_subplot(gs[0+12*k:10+12*k,40],sharey = ax_main[1])
    ax_Spectral = [1,2,3,4]
    ax_Spectral[0] = fig.add_subplot(gs[10+12*k,0:10],sharex=ax_main[0])
    ax_Spectral[1] = fig.add_subplot(gs[10+12*k,10:20],sharex=ax_main[0])
    ax_Spectral[2] = fig.add_subplot(gs[10+12*k,20:30],sharex=ax_main[0])
    ax_Spectral[3] = fig.add_subplot(gs[10+12*k,30:40],sharex=ax_main[0])
    fig.subplots_adjust(wspace = 0,hspace=0)

    # Set up the tics
    ax_main[1].set_yticks([])
    ax_main[2].set_yticks([])
    ax_main[3].set_yticks([])
    ax_Spectral[0].set_yticks([])
    ax_Spectral[1].set_yticks([])
    ax_Spectral[2].set_yticks([])
    ax_Spectral[3].set_yticks([])
    ax_NN.set_xticks([])

    for p,model in enumerate(models):
        # Load the data
        f2 = f"data/{folder}/{classes}_NN.txt"
        if DC:
            f1 = f"data/{folder}/{classes}_DC_{model}.txt"
            f3 = f"data/{folder}/{classes}_DC_Spectral.txt"
            f4 = f"data/{folder}/{classes}_DC_GraphTool.txt"

        else:
            f1 = f"data/{folder}/{classes}_{model}.txt"
            f3 = f"data/{folder}/{classes}_Spectral.txt"
            f4 = f"data/{folder}/{classes}_GraphTool.txt"
            
            

        test_accs = np.genfromtxt(f1)
        test_accs_nn = np.genfromtxt(f2)
        test_accs_spectral = np.genfromtxt(f3)

        # Obtain GNN data
        z = test_accs[:,0] # accuracies
        x = test_accs[:,1] # lambda
        y = test_accs[:,2] # mu

        print(classes,model)
        # appropriately shape the data
        row_size = 121
        z = z.reshape(200,row_size)
        x = x.reshape(200,row_size)
        y = y.reshape(200,row_size)

        # Obtain NN and Spectral data
        NN_c = test_accs_nn[:,0]
        NN_y = test_accs_nn[:,1]
        NN_x = test_accs_nn[:,2]
        S_x = test_accs_spectral[:,2]
        S_c = test_accs_spectral[:,0]
        S_y = test_accs_spectral[:,1]

        test_accs_tool = np.genfromtxt(f4)
        T_x = test_accs_tool[:,2]
        T_c = test_accs_tool[:,0]
        T_y = test_accs_tool[:,1]

        # Smoothen the data with a gaussian filter
        kernel = np.array([[1,4,7,4,1],
                        [4,16,26,16,4],
                        [7,26,41,26,7],
                        [4,16,26,16,4],
                        [1,4,7,4,1]])
        z = convolve(z,kernel)/kernel.sum()
        z = convolve(z,kernel)/kernel.sum()


        # Plot the data
        im_main = ax_main[p].scatter(x,y,c=z,cmap="coolwarm",vmin=1/classes,vmax=1)
        ax_Spectral[p].scatter(S_y,S_x,c=S_c,cmap="coolwarm",vmin=1/classes,vmax=1)
        # ax_Spectral[p].plot([-3,0],[0,0],"--k")
        ax_Spectral[p].scatter(T_y,T_x,c=T_c,cmap="coolwarm",vmin=1/2,vmax=1)


        # Set up Colorbar
        plt.colorbar(im_main, ax=ax_main,cax = fig.add_subplot(gs[0+12*k:10+12*k,42]))
        if k == 0:
            ax_main[p].set_title(f"{model}")
        
        # Set up various labels and limits
        ax_main[p].set_xticks([])
        ax_main[p].set_ylim(0,1)
        ax_main[0].set_ylabel(f"{classes} Classes\nFeature Information")
        ax_NN.set_title(f"NN Accuracy")
        ax_Spectral[0].set_ylabel("Graph\nBased\nAccuracy\n\n")
        ax_main[0].set_yticks([0,.5,1])

    ax_NN.scatter(NN_y,NN_x,c=NN_c,cmap="coolwarm",vmin=1/classes,vmax=1)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

# then we plot class sizes of 5 and 7

# Set up the grid and parameters
gs = GridSpec(23,43)
fig = plt.figure()
num_class = [5,7]
for k,classes in enumerate(num_class):
    # Set up parameters
    models = ["GCN","GAT","SAGE","Transformer"]

    # Add subplots for GNN
    ax_main = [1,2,3,4]
    ax_main[0] = fig.add_subplot(gs[0+12*k:10+12*k,0:10])
    ax_main[1] = fig.add_subplot(gs[0+12*k:10+12*k,10:20])
    ax_main[2] = fig.add_subplot(gs[0+12*k:10+12*k,20:30],sharey = ax_main[1])
    ax_main[3] = fig.add_subplot(gs[0+12*k:10+12*k,30:40],sharey = ax_main[1])

    # Add subplots for NN and Spectral clustering
    ax_NN = fig.add_subplot(gs[0+12*k:10+12*k,40],sharey = ax_main[1])
    ax_Spectral = [1,2,3,4]
    ax_Spectral[0] = fig.add_subplot(gs[10+12*k,0:10],sharex=ax_main[0])
    ax_Spectral[1] = fig.add_subplot(gs[10+12*k,10:20],sharex=ax_main[0])
    ax_Spectral[2] = fig.add_subplot(gs[10+12*k,20:30],sharex=ax_main[0])
    ax_Spectral[3] = fig.add_subplot(gs[10+12*k,30:40],sharex=ax_main[0])
    fig.subplots_adjust(wspace = 0,hspace=0)

    # Set up the tics
    ax_main[1].set_yticks([])
    ax_main[2].set_yticks([])
    ax_main[3].set_yticks([])
    ax_Spectral[0].set_yticks([])
    ax_Spectral[1].set_yticks([])
    ax_Spectral[2].set_yticks([])
    ax_Spectral[3].set_yticks([])
    ax_NN.set_xticks([])

    for p,model in enumerate(models):
        # Load the data
        f2 = f"data/{folder}/{classes}_NN.txt"
        if DC:
            f1 = f"data/{folder}/{classes}_DC_{model}.txt"
            f3 = f"data/{folder}/{classes}_DC_Spectral.txt"
            f4 = f"data/{folder}/{classes}_DC_GraphTool.txt"

        else:
            f1 = f"data/{folder}/{classes}_{model}.txt"
            f3 = f"data/{folder}/{classes}_Spectral.txt"
            f4 = f"data/{folder}/{classes}_GraphTool.txt"

        test_accs = np.genfromtxt(f1)
        test_accs_nn = np.genfromtxt(f2)
        test_accs_spectral = np.genfromtxt(f3)

        # Obtain GNN data
        z = test_accs[:,0] #accs
        x = test_accs[:,1]# lambda
        y = test_accs[:,2]# mu

        # appropriately shape the data
        row_size = 121
        z = z.reshape(200,row_size)
        x = x.reshape(200,row_size)
        y = y.reshape(200,row_size)

        # Obtain NN and Spectral data
        NN_x = test_accs_nn[:,2]
        NN_c = test_accs_nn[:,0]
        NN_y = test_accs_nn[:,1]
        S_x = test_accs_spectral[:,2]
        S_c = test_accs_spectral[:,0]
        S_y = test_accs_spectral[:,1]

        test_accs_tool = np.genfromtxt(f4)
        T_x = test_accs_tool[:,2]
        T_c = test_accs_tool[:,0]
        T_y = test_accs_tool[:,1]

        # Smoothen the data with a gaussian filter
        kernel = np.array([[1,4,7,4,1],
                        [4,16,26,16,4],
                        [7,26,41,26,7],
                        [4,16,26,16,4],
                        [1,4,7,4,1]])
        z = convolve(z,kernel)/kernel.sum()
        z = convolve(z,kernel)/kernel.sum()

        # Plot the data
        im_main = ax_main[p].scatter(x,y,c=z,cmap="coolwarm",vmin=1/classes,vmax=1)
        ax_Spectral[p].scatter(S_y,S_x,c=S_c,cmap="coolwarm",vmin=1/classes,vmax=1)
        # ax_Spectral[p].plot([-3,0],[0,0],"--k")
        ax_Spectral[p].scatter(T_y,T_x,c=T_c,cmap="coolwarm",vmin=1/2,vmax=1)

        plt.colorbar(im_main, ax=ax_main,cax = fig.add_subplot(gs[0+12*k:10+12*k,42]))

        # Set up various labels and limits
        if k == 0:
            ax_main[p].set_xticks([])
        ax_main[p].set_ylim(0,1)
        ax_main[0].set_ylabel(f"{classes} Classes\nFeature Information")
        ax_NN.set_title(f"NN Accuracy")
        ax_Spectral[0].set_ylabel("Graph\nBased\nAccuracy\n\n")
        ax_main[0].set_yticks([0,.5,1])
        if k == 1:
            ax_Spectral[p].set_xlabel("Edge Information")

    ax_NN.scatter(NN_y,NN_x,c=NN_c,cmap="coolwarm",vmin=1/classes,vmax=1)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()
