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

# We show the accuracy plots first

# Set up various hyperparameters
num_classes = 7
DC = False
models = ["GCN","GAT","SAGE","Transformer"]
sns.set()
palette = sns.color_palette("colorblind")

# Create subplots to hold figures
fig = plt.figure()
gs = GridSpec(22,43)
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[0:10,0:10])
ax_main[1] = fig.add_subplot(gs[0:10,10:20])
ax_main[2] = fig.add_subplot(gs[0:10,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[0:10,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)

# Loop through all of the models
for p,model in enumerate(models):
    # Set which files you are pulling from
    f2 = f"data/maxed_runs/{num_classes}_NN.txt"
    if DC: 
        f1 = f"data/maxed_runs/{num_classes}_DC_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_DC_Spectral.txt"
    else:
        f1 = f"data/maxed_runs/{num_classes}_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_Spectral.txt"

    # First we retreive all of the values from the txt files
    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)

    # Assign GNN values
    z = test_accs[:,0] # accuracies
    x = test_accs[:,1] # lambda
    y = test_accs[:,2 ]# mu
    
    # reshape variables to be in the form of a grid
    row_size = 121
    z = z.reshape(200,row_size)
    x = x.reshape(200,row_size)
    y = y.reshape(200,row_size)

    # Assign NN values
    NN_x = test_accs_nn[:,2]
    NN_c = test_accs_nn[:,0]
    NN_y = test_accs_nn[:,1]

    # Spectral Clustering Values
    S_x = test_accs_spectral[:,2]
    S_c = test_accs_spectral[:,0]
    S_y = test_accs_spectral[:,1]

    # Smoothen the plot with a gaussian kernel
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # Create the main plot, with the minimum color being random guessing
    
    im_main = ax_main[p].scatter(x,y,c=z,cmap="coolwarm",vmin=1/num_classes,vmax=1)
    
    # Add Colorbar
    colorbars = fig.add_subplot(gs[5:15,42])
    plt.colorbar(im_main, ax=ax_main,cax = colorbars)

    # Set titles and various plot properties
    ax_main[p].set_title(f"{num_classes} Class {model}")
    ax_main[0].set_ylabel("Non-degree-corrected\nFeature Information")
    ax_main[p].set_ylim(0,1)
    ax_main[0].set_xticks([])

# Set the rest of the plot properties
ax_main[0].set_yticks([0,.5,1])
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])
ax_main[0].set_xticks([])
ax_main[1].set_xticks([])
ax_main[2].set_xticks([])
ax_main[3].set_xticks([])

# Show the comparison plots below

# Set up the various hyper-parameters
num_classes = 7
DC = False
models = ["GCN","GAT","SAGE","Transformer"]

# Create subplots to hold figures
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[11:21,0:10])
ax_main[1] = fig.add_subplot(gs[11:21,10:20])
ax_main[2] = fig.add_subplot(gs[11:21,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[11:21,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)

for p,model in enumerate(models):
    # Set which files you are pulling from
    f2 = f"data/maxed_runs/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/maxed_runs/{num_classes}_DC_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_DC_Spectral.txt"
        f4 = f"data/maxed_runs/{num_classes}_DC_GraphTool.txt"

    else:
        f1 = f"data/maxed_runs/{num_classes}_{model}.txt"
        f3 = f"data/maxed_runs/{num_classes}_Spectral.txt"
        f4 = f"data/maxed_runs/{num_classes}_GraphTool.txt"


    # First we retreive all of the values from the txt files
    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)
    test_accs_tool = np.genfromtxt(f4)


    # Assign GNN values
    z = test_accs[:,0] # accuracies
    x = test_accs[:,1] # lambda
    y = test_accs[:,2] # mu

    # reshape variables to be in the form of a grid     
    row_size = 121
    z = z.reshape(200,row_size)
    x = x.reshape(200,row_size)
    y = y.reshape(200,row_size)

    # Assign NN valuesSpectral
    NN_x = test_accs_nn[:,2]
    NN_c = test_accs_nn[:,0]
    NN_y = test_accs_nn[:,1]

    # Spectral Clustering Values
    S_x = test_accs_spectral[:,2]
    S_c = test_accs_spectral[:,0]
    S_y = test_accs_spectral[:,1]

    T_x = test_accs_tool[:,2]
    T_c = test_accs_tool[:,0]
    T_y = test_accs_tool[:,1]

    # Smoothen the plot with a gaussian kernel
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # Generate a comparison matrix between the GNN and NN
    new_Z = np.zeros((200,row_size))
    for i in range(200):
        new_Z[i] = z[i] - test_accs_nn[i][0]
    compare_NN= new_Z

    # Generate a comparison matrix between the GNN and Spectral Clustering
    new_Z = np.zeros((121,200))
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
    compare_Spectral = new_Z.T

    # Comparing GNN and Tool
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_tool[i%61][0]
    compare_Tool = new_Z.T

    # Smoothen the comparison plots further with a gaussian kernel
    compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
    compare_Tool = convolve(compare_Tool,kernel)/np.sum(kernel)


    # plot region where GNN is better than NN and Spectral
    improvement_threshold = .01
    idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold),(compare_Tool > improvement_threshold))
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="GNN")

    # plot region where NN is better than GNN
    idx = compare_NN < -improvement_threshold
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")


    # plot region where Spectral is better than GNN
    # only measured for positive edge information
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

    # Set various plot properties
    ax_main[p].set_xlabel(f"Edge Information")
    ax_main[p].set_ylim(0,1)
    ax_main[0].legend(bbox_to_anchor=(1.0,0.6),\
    bbox_transform=plt.gcf().transFigure)
    ax_main[0].set_ylabel("Non-degree-corrected\nFeature Information")

# Set up rest of the plot properties
ax_main[0].set_yticks([0,.5,1])
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])

plt.show()
