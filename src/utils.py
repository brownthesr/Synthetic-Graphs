import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
def adj_to_list(a):
    adjList =[[],[]]
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j]== 1:
                adjList[0].append(i)
                adjList[1].append(j)
    return adjList

def accuracy(preds,mask,labels):# obtains the accuracy of the model
    correct = (preds[mask] == labels[mask]).sum()
    acc = int(correct)/int(mask.sum())
    return acc

def draw_graph(adj,communities,feat):
    """
    Draws the graph

    This assigns colors according to the communities. Warning -
    does not work well for graphs of over 500 nodes. Additionally
    parameters only color nodes up to 4 different groups, additional
    coloring can be added. Isolated nodes are removed to enhance 
    visibility.

    Parameters
    ----------
    adj : numpy array of shape (num_nodes, num_nodes)
        The adjacency matrix
    communities : list of size (num_nodes)
        The community assignments
    """
    g = nx.from_numpy_array(adj)
    isos = list(nx.isolates(g))
    mask = np.ones(len(communities), dtype=bool)
    mask[isos] = False
    #print(nx.number_of_isolates(g)) # used for debugging
    communities = communities[mask]
    g.remove_nodes_from(isos)
    g.remove_edges_from(nx.selfloop_edges(g))
    colors = ["yellow"]*400
    colors = np.array(colors)
    colors[np.where(communities == 0)] = "green"
    colors[np.where(communities == 1)] = "blue"
    colors[np.where(communities == 2)] = "red"
    colors[np.where(communities == 3)] = "yellow"
    plt.scatter(feat[:,0],feat[:,1],color=colors)
    plt.show()