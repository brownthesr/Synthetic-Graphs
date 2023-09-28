import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
types = ['Heirarchical','Triadic','Epsilon']
for i in range(3):
    plt.subplot(1,3,i+1)
    gat_file = np.loadtxt(f"data/structure/{types[i]}_sbm_GAT.txt")
    gcn_file = np.loadtxt(f"data/structure/{types[i]}_sbm_GCN.txt")
    sage_file = np.loadtxt(f"data/structure/{types[i]}_sbm_SAGE.txt")

    plt.plot(gat_file[:,-1],gat_file[:,0],color="C0",ls="--",label="GAT Structured")
    plt.plot(gat_file[:,-1],gat_file[:,1],color="C0",label="GAT Unstructured")
    plt.plot(gcn_file[:,-1],gcn_file[:,0],color="C1",ls="--",label="GCN Structured")
    plt.plot(gcn_file[:,-1],gcn_file[:,1],color="C1",label="GCN Unstructured")
    plt.plot(sage_file[:,-1],sage_file[:,0],color="C2",ls="--",label="SAGE Structured")
    plt.plot(sage_file[:,-1],sage_file[:,1],color="C2",label="SAGE Unstructured")
    plt.title(f"{types[i]} Structure")
    plt.ylim(0,1)
    plt.xlim(0,1)

plt.legend( loc='upper left', bbox_to_anchor=(1, 1))
# plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.show()