import matplotlib.pyplot as plt
import numpy as np
plt.figure()

compare = False# this is for the comparison of the two methods

test_accs = np.genfromtxt("mu_lambda_variations_GCN.txt")
test_accs_vanilla = np.genfromtxt("mu_lambda_variations_NN.txt")



x = test_accs[:,1]# lambda
z = test_accs[:,0] #accs
z = z.reshape(200,61)
x = x.reshape(200,61)
y = test_accs[:,2]# mu

y = y.reshape(200,61)
new_Z = np.zeros((200,61))
for i in range(200):
    new_Z[i] = z[i] - test_accs_vanilla[i][0]
if compare:
    z = new_Z

plt.xlabel("edge info - normalized degree separation")# normalized degree separation, lambda
plt.ylabel("feature info - cloud distance from origin")#  mean featue separation, mu#

colors = []
for a in z:

    colors.append((2*a-1,2-2*a,.5))
plt.scatter(x,y,c=z,cmap="coolwarm")
plt.colorbar()
if compare:
    plt.clim(-.4,.4)

plt.show()
