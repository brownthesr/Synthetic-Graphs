"""This file is specifically made to generate a graph that compares all 4 of the models
side by side. It compares across all number of classes, additionally it uses degree corrected
distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.spatial import Delaunay
import numpy as np
from scipy.ndimage import convolve
import seaborn as sns



maxes = True
if maxes:
    folder = "maxed_runs"
else:
    folder = "averaged_runs/Masters"
# Two Classes
# Defining various hyperparameters
num_classes = 2
DC = True
models = ["GCN","GAT","SAGE","Transformer"]
sns.set()
palette = sns.color_palette("colorblind")

# Creating figures for subplots
fig = plt.figure()
gs = GridSpec(44,43)
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[0:10,0:10])
ax_main[1] = fig.add_subplot(gs[0:10,10:20])
ax_main[2] = fig.add_subplot(gs[0:10,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[0:10,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)

for p,model in enumerate(models):
    # Load the data
    f2 = f"data/{folder}/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/{folder}/{num_classes}_DC_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_DC_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_DC_GraphTool.txt"

    else:
        f1 = f"data/{folder}/{num_classes}_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_GraphTool.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)

    # Obtaining Data for GNN
    z = test_accs[:,0] # accs
    x = test_accs[:,1] # lambda
    y = test_accs[:,2] # mu

    # Reshape the data
    row_size = 121
    z = z.reshape(200,row_size)
    x = x.reshape(200,row_size)
    y = y.reshape(200,row_size)

    # Obtaining Data for NN and Spectral Clustering
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
    

    # Smoothening the data with a gaussian filter
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # Comparing the GNN performance to that of the NN
    new_Z = np.zeros((200,row_size))
    for i in range(200):
        new_Z[i] = z[i] - test_accs_nn[i][0]
    compare_NN= new_Z

    # Comparing the GNN performance to that of the Spectral Clustering
    new_x = x[:,:]
    new_y = y[:,:]
    new_Z = np.zeros((121,200))
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
    compare_Spectral = new_Z.T

    # Smoothening the data with a gaussian filter
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
    compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)

    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_tool[i%61][0]
    compare_Tool = new_Z.T
    compare_Tool = convolve(compare_Tool,kernel)/np.sum(kernel)

    # plotting region where the GNN is better than the NN and Spectral Clustering
    improvement_threshold = .01
    idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold), (compare_Tool > improvement_threshold))
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="GCN")

    # plotting region where the GNN is worse than NN
    idx = compare_NN < -improvement_threshold
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")

    # plotting region where the GNN is worse than spectral clustering
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
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[8],label="Graph\nBased")

    # Plot the region where tool outperforms GNN
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

    # Set the title and labels
    ax_main[p].set_title(f"{model}")
    ax_main[p].set_ylim(0,1)
    ax_main[0].set_ylabel(f"{num_classes} class\nFeature\nInformation")

# fixing ticks on plots
ax_main[0].set_yticks([0,.5,1])
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])
ax_main[0].set_xticks([])
ax_main[1].set_xticks([])
ax_main[2].set_xticks([])
ax_main[3].set_xticks([])


# Three Classes
# Defining various parameters
num_classes = 3
models = ["GCN","GAT","SAGE","Transformer"]
ax_main = [1,2,3,4]

# creating the figure with subplots
ax_main[0] = fig.add_subplot(gs[11:21,0:10])
ax_main[1] = fig.add_subplot(gs[11:21,10:20])
ax_main[2] = fig.add_subplot(gs[11:21,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[11:21,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)

for p,model in enumerate(models):
    # Loading the data
    f2 = f"data/{folder}/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/{folder}/{num_classes}_DC_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_DC_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_DC_GraphTool.txt"

    else:
        f1 = f"data/{folder}/{num_classes}_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_GraphTool.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)
    
    # Obtaining the data for GNN
    z = test_accs[:,0] #accs
    x = test_accs[:,1]# lambda
    y = test_accs[:,2]# mu

    # Reshaping the data
    row_size = 121
    z = z.reshape(200,row_size)
    x = x.reshape(200,row_size)
    y = y.reshape(200,row_size)

    # Obtaining the data for NN and Spectral Clustering
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

    # Smoothening the data with a gaussian filter
    kernel = np.array([[1,4,7,4,1],
                    [4,16,26,16,4],
                    [7,26,41,26,7],
                    [4,16,26,16,4],
                    [1,4,7,4,1]])
    z = convolve(z,kernel)/kernel.sum()
    z = convolve(z,kernel)/kernel.sum()

    # comparing the GNN to the NN
    new_Z = np.zeros((200,row_size))
    for i in range(200):
        new_Z[i] = z[i] - test_accs_nn[i][0]
    compare_NN= new_Z

    # comparing the GNN to Spectral Clustering
    new_x = x[:,:]
    new_y = y[:,:]
    new_Z = np.zeros((121,200))
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
        #print(test_accs_vanilla[i][0])
    compare_Spectral = new_Z.T

    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_tool[i%61][0]
    compare_Tool = new_Z.T
    compare_Tool = convolve(compare_Tool,kernel)/np.sum(kernel)

    # Smoothening the data with a gaussian filter
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
    compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)

    # plotting where GNN is better than NN and Spectral Clustering
    improvement_threshold = .01
    idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold), (compare_Tool > improvement_threshold))
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="GNN")

    # plotting where NN is better than GNN
    idx = compare_NN < -improvement_threshold
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")

    # comparing the GNN to Spectral Clustering and plotting where Spectral Clustering is better than GNN
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

    # Plot the region where tool outperforms GNN
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

    # setting the labels
    ax_main[p].set_ylim(0,1)
    ax_main[0].legend(bbox_to_anchor=(.95,0.6),\
    bbox_transform=plt.gcf().transFigure)
    ax_main[0].set_ylabel(f"{num_classes} class\nFeature\nInformation")

# fixing ticks on the plots
ax_main[0].set_xticks([])
ax_main[1].set_xticks([])
ax_main[2].set_xticks([])
ax_main[3].set_xticks([])
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])
ax_main[0].set_yticks([0,.5,1])


# 5 classes

# setting up various hyperparameters
num_classes = 5
models = ["GCN","GAT","SAGE","Transformer"]

# setting up the figure
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[22:32,0:10])
ax_main[1] = fig.add_subplot(gs[22:32,10:20])
ax_main[2] = fig.add_subplot(gs[22:32,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[22:32,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])

# Looping over the models
for p,model in enumerate(models):
    # loading the data
    f2 = f"data/{folder}/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/{folder}/{num_classes}_DC_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_DC_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_DC_GraphTool.txt"

    else:
        f1 = f"data/{folder}/{num_classes}_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_GraphTool.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)
    
    # setting up the data
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

    test_accs_tool = np.genfromtxt(f4)
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

    # comparing the GNN to NN and plotting where GNN is better than NN
    new_Z = np.zeros((200,row_size))
    for i in range(200):
        new_Z[i] = z[i] - test_accs_nn[i][0]
    compare_NN= new_Z

    # comparing the GNN to Spectral Clustering and plotting where GNN is better than Spectral Clustering
    new_Z = np.zeros((121,200))
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
    compare_Spectral = new_Z.T

    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_tool[i%61][0]
    compare_Tool = new_Z.T
    compare_Tool = convolve(compare_Tool,kernel)/np.sum(kernel)

    # Smoothening the data with a gaussian filter
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
    compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)

    # setting up the threshold for plotting
    # where the GNN is better than the NN and Spectral Clustering
    improvement_threshold = .01
    idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold), (compare_Tool > improvement_threshold))

    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="GNN")

    # Plotting where NN is better than GNN
    idx = compare_NN < -improvement_threshold
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")

    # Plotting where Spectral Clustering is better than GNN
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

    # Plot the region where tool outperforms GNN
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

    # Setting up the rest of the plot
    ax_main[p].set_ylim(0,1)
    ax_main[0].legend(bbox_to_anchor=(.95,0.6),\
    bbox_transform=plt.gcf().transFigure)
    ax_main[0].set_ylabel(f"{num_classes} class\nFeature\nInformation")

# setting up other parts of the plot
ax_main[0].set_xticks([])
ax_main[1].set_xticks([])
ax_main[2].set_xticks([])
ax_main[3].set_xticks([])
ax_main[0].set_yticks([0,.5,1])

# 7 classes
# setting up various hyperparameters
num_classes = 7
models = ["GCN","GAT","SAGE","Transformer"]

# setting up the figure
ax_main = [1,2,3,4]
ax_main[0] = fig.add_subplot(gs[33:43,0:10])
ax_main[1] = fig.add_subplot(gs[33:43,10:20])
ax_main[2] = fig.add_subplot(gs[33:43,20:30],sharey = ax_main[1])
ax_main[3] = fig.add_subplot(gs[33:43,30:40],sharey = ax_main[1])
fig.subplots_adjust(wspace = 0,hspace=0)
ax_main[1].set_yticks([])
ax_main[2].set_yticks([])
ax_main[3].set_yticks([])

# Looping through the different models
for p,model in enumerate(models):
    # Loading the data
    f2 = f"data/{folder}/{num_classes}_NN.txt"
    if DC:
        f1 = f"data/{folder}/{num_classes}_DC_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_DC_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_DC_GraphTool.txt"

    else:
        f1 = f"data/{folder}/{num_classes}_{model}.txt"
        f3 = f"data/{folder}/{num_classes}_Spectral.txt"
        f4 = f"data/{folder}/{num_classes}_GraphTool.txt"

    test_accs = np.genfromtxt(f1)
    test_accs_nn = np.genfromtxt(f2)
    test_accs_spectral = np.genfromtxt(f3)

    # Setting up the data
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

    test_accs_tool = np.genfromtxt(f4)
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

    # Comparing the GNN to the NN and Spectral Clustering
    new_Z = np.zeros((200,row_size))
    for i in range(200):
        new_Z[i] = z[i] - test_accs_nn[i][0]
    compare_NN= new_Z

    new_x = x[:,:]
    new_y = y[:,:]
    new_Z = np.zeros((121,200))
    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
    compare_Spectral = new_Z.T

    # Smoothening the data with a gaussian filter
    compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
    compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)

    for i in range(121):
        new_Z[i] = z[:,i] - test_accs_tool[i%61][0]
    compare_Tool = new_Z.T
    compare_Tool = convolve(compare_Tool,kernel)/np.sum(kernel)

    # plotting where GNN is better than NN and Spectral Clustering
    improvement_threshold = .01
    idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold), (compare_Tool > improvement_threshold))
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="GNN")

    # plotting where NN is better than GNN
    idx = compare_NN < -improvement_threshold
    xpoints=x[idx]
    ypoints=y[idx]
    points = np.vstack([xpoints,ypoints]).T
    sample = points
    ax_main[p].scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")

    # plotting where Spectral Clustering is better than GNN
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

    # Plot the region where tool outperforms GNN
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



    # plotting the rest of the data
    ax_main[p].set_xlabel(f"Edge Information")
    ax_main[p].set_ylim(0,1)
    ax_main[0].legend(bbox_to_anchor=(.95,0.6),\
    bbox_transform=plt.gcf().transFigure)
    ax_main[0].set_ylabel(f"{num_classes} class\nFeature\nInformation")

# setting up rest of parameters
ax_main[0].set_yticks([0,.5,1])

plt.show()
