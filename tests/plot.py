"""
Plots the results of our data tests
"""
import matplotlib.pyplot as plt
import numpy as np
plt.figure()

COMPARE = False# this is for the comparison of the two methods
NN = True
EIG = False
test_accs = np.genfromtxt("data/averaged_runs/SAGE.txt")
test_accs_vanilla = np.genfromtxt("data/averaged_runs/NN.txt")
#print(len(test_accs))
print(test_accs.shape)
#note put the NN and the eigenvector in test_accs_vanilla
x = test_accs[:,1]# lambda
z = test_accs[:,0] #accs
z = z.reshape(200,61)
x = x.reshape(200,61)
y = test_accs[:,2]# mu
first_z = z
#y = y.reshape(200,61)
print(test_accs_vanilla.shape)

if COMPARE:
    new_Z = np.zeros((200,61))
    if NN:
        for i in range(200):
            new_Z[i] = z[i] - test_accs_vanilla[i][0]
    elif EIG:
        new_Z = np.zeros((61,200))
        for i in range(61):
            new_Z[i] = z[:,i] - test_accs_vanilla[i][0]
            #print(test_accs_vanilla[i][0])
        new_Z = new_Z.T
    else:
        test_accs_vanilla = test_accs_vanilla.reshape(200,61,3)
        for i in range(200):
            for j in range(len(test_accs_vanilla[i])):
                new_Z[i][j] = z[i,j]-test_accs_vanilla[i][j][0]
    z = new_Z


colors = []
for a in z:

    colors.append((2*a-1,2-2*a,.5))
if COMPARE:
    fig,ax = plt.subplots(1,3)
    if NN:
        comp = ax[0].scatter(x,y,c=z,cmap="coolwarm",vmin = -.4,vmax=.4)
        plt.colorbar(comp,ax=ax[0])
        ax[2].scatter(test_accs_vanilla[:,2],test_accs_vanilla[:,0])
        normal = ax[1].scatter(x,y,c=first_z,cmap="coolwarm",vmin=.5,vmax=1)
        plt.colorbar(normal, ax=ax[1])
        ax[0].set_title("Comparison of Two")
        ax[1].set_title("Acc of GCN/GAT/SAGE")
        ax[2].set_title("Acc of NN")
    elif EIG:
        comp = ax[1].scatter(x,y,c=first_z,cmap="coolwarm",vmin=.5,vmax=1)
        comp_color = plt.colorbar(comp,ax = ax[1])
        normal = ax[0].scatter(x,y,c=z,cmap="coolwarm",vmin=-.4,vmax = .4)
        plt.colorbar(normal,ax=ax[0])
        ax[2].scatter(test_accs_vanilla[:,1],test_accs_vanilla[:,0])
        ax[0].set_title("Comparison of Two")
        ax[1].set_title("Acc of GCN/GAT/SAGE")
        ax[2].set_title("Acc of Spectral")
    else:
        comp = ax[1].scatter(x,y,c=first_z,cmap="coolwarm",vmin=.5,vmax=1)
        comp_color = plt.colorbar(comp,ax = ax[1])
        normal = ax[0].scatter(x,y,c=z,cmap="coolwarm",vmin=-.4,vmax = .4)
        plt.colorbar(normal,ax=ax[0])
        other = ax[2].scatter(test_accs_vanilla[:,:,1],test_accs_vanilla[:,:,2],c=test_accs_vanilla[:,:,0],cmap="coolwarm",vmin = .5,vmax = 1)
        plt.colorbar(other,ax=ax[2])
        ax[0].set_title("Comparison of Two")
        ax[1].set_title("Acc of GCN/GAT/SAGE")
        ax[2].set_title("Acc of Second GCN/GAT/SAGE")
    
else:
    plt.scatter(x,y,c=z,cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("edge info - interpolation between random and planted")
    # normalized degree separation, lambda
    plt.ylabel("feature info - cloud distance from origin")#  mean featue separation, mu#
# if COMPARE:
#     plt.clim(-.4,.4)

#plt.ylim(0,6)
plt.show()
