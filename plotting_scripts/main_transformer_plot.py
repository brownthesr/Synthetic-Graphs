"""This is where we plot the main binary GCN plot
"""

from telnetlib import XASCII
from tkinter.messagebox import YES
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.spatial import Delaunay
import seaborn as sns
import numpy as np
from scipy.ndimage import convolve

# Here we plot the accuracies for the binary GCN model
num_classes = 2
DC = False
model = "Transformer"
show_positive = True
show_negative = True
sns.set()
palette = sns.color_palette("colorblind")

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

# Setting up the figure
fig = plt.figure()
gs = GridSpec(11,50)
ax_main = fig.add_subplot(gs[0:10,0:20])
ax_NN = fig.add_subplot(gs[:10,20:22])
ax_Spectral = fig.add_subplot(gs[10,0:20],sharex=ax_main)
ax_compare = fig.add_subplot(gs[:10,30:50],sharey=ax_main)
fig.subplots_adjust(wspace = 0,hspace=0)
# Setting up the x-ticks
ax_NN.set_xticks([])
ax_Spectral.set_yticks([])

# Smoothing the data with a gaussian filter
kernel = np.array([[1,4,7,4,1],
                   [4,16,26,16,4],
                   [7,26,41,26,7],
                   [4,16,26,16,4],
                   [1,4,7,4,1]])
z = convolve(z,kernel)/kernel.sum()
z = convolve(z,kernel)/kernel.sum()

# Comparing GNN and NN
new_Z = np.zeros((200,row_size))
for i in range(200):
    new_Z[i] = z[i] - test_accs_nn[i][0]
compare_NN= new_Z
    
# Comparing GNN and Spectral
new_Z = np.zeros((121,200))
for i in range(121):
    new_Z[i] = z[:,i] - test_accs_spectral[i%61][0]
    #print(test_accs_vanilla[i][0])
compare_Spectral = new_Z.T

# Comparing GNN and Tool


# Smoothing the data with a gaussian filter
compare_Spectral = convolve(compare_Spectral,kernel)/np.sum(kernel)
compare_NN = convolve(compare_NN,kernel)/np.sum(kernel)

for i in range(121):
    new_Z[i] = z[:,i] - test_accs_tool[i%61][0]
compare_Tool = new_Z.T
compare_Tool = convolve(compare_Tool,kernel)/np.sum(kernel)

# Plot the region where GNN outperforms NN and Spectral
improvement_threshold = .01
idx = np.logical_and((compare_NN > improvement_threshold), (compare_Spectral > improvement_threshold), (compare_Tool > improvement_threshold))
xpoints=x[idx]
ypoints=y[idx]
points = np.vstack([xpoints,ypoints]).T
sample = points
ax_compare.scatter(x=sample[:,0],y=sample[:,1],color=palette[1],label="Transformer")

# Plot the region where NN outperforms GNN 
idx = compare_NN < -improvement_threshold
xpoints=x[idx]
ypoints=y[idx]
points = np.vstack([xpoints,ypoints]).T
sample = points
ax_compare.scatter(x=sample[:,0],y=sample[:,1],color=palette[2],label="NN")

# Plot the region where Spectral outperforms GNN
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
ax_compare.scatter(x=sample[:,0],y=sample[:,1],color=palette[8],label="Graph\nBased")

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
ax_compare.scatter(x=sample[:,0],y=sample[:,1],color=palette[8],label="_Graph\nBased")

# Puts all the regions in the figures
im_main = ax_main.scatter(x,y,c=z,cmap="coolwarm",vmin=1/2,vmax=1)
ax_NN.scatter(NN_y,NN_x,c=NN_c,cmap="coolwarm",vmin=1/2,vmax=1)
ax_Spectral.scatter(S_y,S_x,c=S_c,cmap="coolwarm",vmin=1/2,vmax=1)
ax_Spectral.scatter(T_y,T_x,c=T_c,cmap="coolwarm",vmin=1/2,vmax=1)

# Sets x and y limits
ax_main.set_xlim(-3.1,3.1)
ax_main.set_ylim(0,1)
ax_main.set_yticks([.25,.5,.75,1])
ax_NN.set_ylim(0,1)
ax_NN.set_yticks([])
ax_compare.set_ylim(0,1)
ax_compare.set_xlim(-3.1,3.1)

# Title the spectral clustering and plot negative clustering
ax_Spectral.set_ylabel("Graph Based\n Clustering\n Accuracy")
# ax_Spectral.plot([-3,0],[0,0],"--k")
# ax_Graphtool.set_ylabel("GraphTool\n Clustering\n Accuracy")
# ax_Graphtool.plot([0,3],[0,0],"--k")
# ax_Graphtool.set_ylim(-.1,.1)

# set up colorbar
colorbar =  fig.add_subplot(gs[0:6,24])
colorbar.set_yticks([.5,.75,1])
plt.colorbar(im_main, ax=ax_main,cax = colorbar)

# Set titles and labels
ax_main.set_title(f"{num_classes} Class {model} Accuracy")
ax_main.grid(color="white")
ax_main.set_ylabel("Feature Information")
ax_NN.set_title(f"NN Accuracy")
ax_compare.set_title("Favorable Regimes")
ax_compare.set_xlabel("Edge Information")
ax_compare.set_ylabel("Feature Information")
colorbar.set_title("Key")
ax_Spectral.set_xlabel("Edge Information")
ax_compare.legend()

plt.show()
