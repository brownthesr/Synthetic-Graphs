import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility

fig = plt.figure()
ax = plt.axes(projection="3d")

rows = 50
cols = 8
num_bars = rows*cols
x_pos = np.linspace(.14,.9,cols).tolist()*rows
y_pos = np.repeat(np.arange(rows),cols)
z_pos = [0] * num_bars
x_size = np.ones(num_bars)/8/1.5
y_size = np.ones(num_bars)/1.5
z_size = np.genfromtxt("sbm_gnn_transfer.txt")
ax.set_xlabel("initial run accuracy")
ax.set_ylabel("differing random graphs")
ax.set_zlabel("overall accuracy")

order_all = True
order_biggest = False
#z_size = z_size[:rows]
if(order_all):
    for i in range(cols):
        ordering = z_size[:,i]
        idx = np.flip(np.argsort(ordering))
        z_size[:,i] = z_size[idx,i]
if order_biggest:
    ordering = z_size[:,7]
    idx = np.flip(np.argsort(ordering))
    z_size = z_size[idx]
averages = np.sum(z_size,axis = 0)/200
print(averages)
start = 0*8

z_size = z_size.flatten()[:num_bars]
colors = ["r","k","b","y","g","m","c","orange"]*rows
#change = z_size[0,:8]
#print(change)
#idx = np.argsort(change)
#z_size[0,:8] = change[idx]
#print(z_size[0],idx,change)
#np.savetxt("sbm_gnn_transfer.txt",z_size)
ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size, color=colors)

plt.show()