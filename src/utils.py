"""
This module contains helpful functions to aid in readability
and mundane tasks
"""
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
def adj_to_list(adj):
    """
    Converts adjacency matrix into an edge list
    """
    adj_list =[[],[]]
    for i, row in enumerate(adj):
        for j, item in enumerate(row):
            if item== 1:
                adj_list[0].append(i)
                adj_list[1].append(j)
    return adj_list

def accuracy(preds,mask,labels):
    """
    Obtains the accuracy of a model
    """
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
    graph = nx.from_numpy_array(adj)
    isos = list(nx.isolates(graph))
    mask = np.ones(len(communities), dtype=bool)
    mask[isos] = False
    #print(nx.number_of_isolates(g)) # used for debugging
    communities = communities[mask]
    graph.remove_nodes_from(isos)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    adj = nx.to_numpy_array(graph)
    colors = ["yellow"]*len(adj)
    colors = np.array(colors)
    colors[np.where(communities == 0)] = "green"
    colors[np.where(communities == 1)] = "blue"
    colors[np.where(communities == 2)] = "red"
    colors[np.where(communities == 3)] = "yellow"
    nx.draw(graph,node_color=colors)
    plt.show()
