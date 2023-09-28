"""This file is specifically made to generate a graph that compares all 4 of the models
side by side. Then it will compare Degree-corrected vs poisson
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


# Plotting non-degree corrected data
# Setting up various hyperparameters for drawing the graph
num_classes = 2
DC = False
models = ["GCN","GAT","SAGE","Transformer"]
sns.set()
palette = sns.color_palette("colorblind")

# setting up the figure
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

    # Smoothening the data with a gaussian filter
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # Compare GNN with NN
    new_Z = np.zeros((200,row_size))
    for i in range(200):
        new_Z[i] = z[i] - test_accs_nn[i][0]
    compare_NN= new_Z

    # Compare GNN with Spectral
    new_Z = np.zeros((121,200))
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
        #print(test_accs_vanilla[i][0])
    compare_Spectral = new_Z.T

    # Smoothening the data with a gaussian filter
    compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)

    # plotting where GNN outperforms NN and Spectral
    improvement_threshold = .01
    idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold))
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="GCN")

    # plotting where GNN underperforms NN
    idx = compare_NN < -improvement_threshold
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")

    # plotting where GNN underperforms Spectral
    new_x = x[:,60:]
    new_y = y[:,60:]
    new_Z = np.zeros((61,200))
    for i in range(61):
        new_Z[i] = z[:,60+i] - test_accs_spectral[i][0]
    compare_Spectral = new_Z.T
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
    idx = compare_Spectral < -improvement_threshold
    xpoints=new_x[idx]
    ypoints=new_y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[8],label="Graph\nBased")

    # setting up the labels
    ax_main[p].set_title(f"{num_classes} class {model}")
    ax_main[p].set_ylim(0,1)
    ax_main[0].set_ylabel("Non-Degree-CorrectedC\nFeature Information")

# setting up the xtics
ax_main[0].set_yticks([0,.5,1])
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])
ax_main[0].set_xticks([])
ax_main[1].set_xticks([])
ax_main[2].set_xticks([])
ax_main[3].set_xticks([])

# Plotting degree corrected data
num_classes = 2
DC = True
models = ["GCN","GAT","SAGE","Transformer"]

# Setting up the subplots
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[11:21,0:10])
ax_main[1] = fig.add_subplot(gs[11:21,10:20])
ax_main[2] = fig.add_subplot(gs[11:21,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[11:21,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])

# Looping over the models
for p,model in enumerate(models):
    # Loading the data
    f2 = f"data/maxed_runs/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/maxed_runs/{num_classes}_DC_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_DC_Spectral.txt"
        f4 = f"data/maxed_runs/{num_classes}_DC_GraphTool.txt"

    else:
        f1 = f"data/maxed_runs/{num_classes}_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_Spectral.txt"
        f4 = f"data/maxed_runs/{num_classes}_GraphTool.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)
    test_accs_tool = np.genfromtxt(f4)
    
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
    T_x = test_accs_tool[:,2]
    T_c = test_accs_tool[:,0]
    T_y = test_accs_tool[:,1]

    # Smoothening the data with a gaussian filter
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # Calculating the difference between the NN and GNN
    new_Z = np.zeros((200,row_size))
    for i in range(200):
        new_Z[i] = z[i] - test_accs_nn[i][0]
    compare_NN= new_Z

    # Calculating the difference between the Spectral and GNN
    new_Z = np.zeros((121,200))
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
    compare_Spectral = new_Z.T

    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_tool[i%61][0]
    compare_Tool = new_Z.T

    # Smoothening the data with a gaussian filter
    compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)

    # plotting where GNN outperforms NN and Spectral
    improvement_threshold = .01
    idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold), (compare_Tool > improvement_threshold))
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="GNN")

    # plotting where GNN underperforms NN
    idx = compare_NN < -improvement_threshold
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")

    # plotting where GNN underperforms Spectral
    new_x = x[:,60:]
    new_y = y[:,60:]
    new_Z = np.zeros((61,200))
    for i in range(61):
        new_Z[i] = z[:,60+i] - test_accs_spectral[i][0]
    compare_Spectral = new_Z.T
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
    idx = compare_Spectral < -improvement_threshold
    xpoints=new_x[idx]
    ypoints=new_y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[8],label="Graph\nBased")

    # Plot the region where Spectral outperforms GNN
    new_x = x[:,:61]
    new_y = y[:,:61]
    new_Z = np.zeros((61,200))
    for i in range(61):
        new_Z[i] = z[:,i] - test_accs_tool[i][0]
        #print(test_accs_vanilla[i][0])
    compare_tool = new_Z.T
    compare_tool = convolve(compare_tool,kernel)/np.sum(kernel)
    idx = compare_tool < -improvement_threshold
    xpoints=new_x[idx]
    ypoints=new_y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[8],label="_Graph\nBased")

    # setting up label
    ax_main[p].set_xlabel(f"Edge Information")
    ax_main[p].set_ylim(0,1)
    ax_main[0].legend(bbox_to_anchor=(.95,0.6),\
    bbox_transform=plt.gcf().transFigure)
    ax_main[0].set_ylabel("Degree-CorrectedC\nFeature Information")

ax_main[0].set_yticks([0,.5,1])

plt.show()
