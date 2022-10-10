import numpy as np
import matplotlib.pyplot as plt
typ = ["feats","both","edges"]

for t in typ:
    arr = np.genfromtxt(f"GCN_scramble/scramble_{t}.txt")
    x = arr[:,0]
    y = arr[:,1]
    z = arr[:,2]
    print(y-x)

    plt.scatter(x,y,label=t)
names = ["Cora","Citeseer","Pubmed","DBLP","Computers","Photo","Flickr","GitHub","FacebookPagePage","LastFMAsia","DeezerEurope"]
empty = [a for a in range(11)]
plt.xticks(x,labels = empty)
plt.grid()
plt.xlim(0,1)
plt.legend()
plt.ylim(0,1)
plt.ylabel("Synthetic data Performance")
plt.xlabel("Real data Performance")
plt.title("Comparison of Synthetic vs Real Data")
plt.plot([0,1],[0,1])
plt.show()
xs = np.zeros(11)
