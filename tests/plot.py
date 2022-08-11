"""
Plots the results of our data tests
"""
import matplotlib.pyplot as plt
import numpy as np
plt.figure()

COMPARE = True# this is for the comparison of the two methods
NN = False
EIG = False
test_accs = np.genfromtxt("data/mu_lambda_variation_GAT.txt")
test_accs_vanilla = np.genfromtxt("data/mu_lambda_variation_GCN.txt")
#print(len(test_accs))
print(test_accs.shape)
#note put the NN and the eigenvector in test_accs_vanilla
x = test_accs[:,1]# lambda
z = test_accs[:,0] #accs
z = z.reshape(200,61)
x = x.reshape(200,61)
y = test_accs[:,2]# mu

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
plt.xlabel("edge info - interpolation between random and planted")
# normalized degree separation, lambda
plt.ylabel("feature info - cloud distance from origin")#  mean featue separation, mu#

colors = []
for a in z:

    colors.append((2*a-1,2-2*a,.5))
plt.scatter(x,y,c=z,cmap="coolwarm")
plt.colorbar()
if COMPARE:
    plt.clim(-.4,.4)

#plt.ylim(0,6)
plt.show()
