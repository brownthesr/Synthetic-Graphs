import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations,combinations,combinations_with_replacement

def generate_SBM(n,connection_matrix,edge_possibility,class_possibility):
    """This creates a random graph based on:
        n : how many node we have
        connection_matrix : the likelihood of a node in cluster i connecting to a node in cluster j
        edge_possibility : the likelihood of an edge existing between two nodes
        class_possibility : the likelihood of belonging to any given class"""
    num_clusters = len(class_possibility)
    cluster_assignments = np.sort(np.random.choice(np.arange(num_clusters),p=class_possibility,size=n))
    cluster_indices = [cluster_assignments == a for a in range(num_clusters)]
    cluster_assignments = np.expand_dims(cluster_assignments,1)
    #clusters = cluster_assignments == cluster_assignments.T
    probabilities = np.random.random((n,n))
    adj = np.zeros((n,n))
    for i in range(num_clusters):

        for j in range(num_clusters):
            mask = np.zeros((n,n),dtype=bool)
            imask = np.zeros((n,n))
            imask[cluster_indices[i]] = ~mask[cluster_indices[i]]
            if(len(cluster_indices[i]) !=0 and len(cluster_indices[j]) != 0):
                jmask = np.zeros((n,n))
                jmask[:,cluster_indices[j]] =  ~mask[:,cluster_indices[j]]
                total_mask = jmask*imask
                probs = probabilities

                probs = probs < (connection_matrix[i,j]*edge_possibility)
                #print(probs)
                probs = (total_mask*1)*probs
                #print(probs)
                adj += probs
    tri = np.tri((n),k=-1)
    adj = adj* tri
    adj += adj.T
    #print(cluster_assignments)
    return adj,cluster_assignments

def generate_SSBM(n,c,p_intra,p_inter):
    """This is similar to the above SBM but in this case it is symmetric"""
    # assign a community to each node
    community = np.repeat(list(range(c)),np.ceil(n/c))
    
    #np.repeat(list to iterate over, how many times to repeat an item)

    #make sure community has size n
    community = community[0:n]
    communities = community.copy()
    # just in case repeat repeated too many

    # make it a collumn vector
    community = np.expand_dims(community,1)

    # generate a boolean matrix indicating whether 
    # two nodes share a community
    # this is a smart way to generate a section graph
    intra = community == community.T
    inter = community != community.T# we can also use np.logical not

    random = np.random.random((n,n))
    tri = np.tri(n,k=-1).astype(bool)

    intergraph = (random < p_intra) * intra * tri# this creates a matrix that only has trues where
                                                # random< p_intra, they are in intra, and along half the matrix
                                                # (if it were the whole matrix it would be double the edges we want)
    intragraph = (random < p_inter) * inter * tri# same thing here
    graph = np.logical_or(intergraph,intragraph)
    graph = graph*1# this converts it to a int tensor
    graph += graph.T
    return graph,communities

def generate_perms(num_classes,num_feat):
    assert num_classes <= 2**num_feat
    dims = np.ceil(np.log2(num_classes))+1
    vecs = []
    #vec1 = np.ones(num_feat)
    #vecs.append(vec1.copy())
    #vec1[-1] = -1
    #vec1[-2] = -1
    #perms = list(set(permutations(vec1)))


    combs = np.array(list(combinations_with_replacement([0,1],num_feat)))
    total = []
    for i in range(len(combs)):
        total.append(set(permutations(combs[i])))
    for i in total:
        for j in i:
            vecs.append(j)
    
    #print(perms)
    #for i in perms:
    #    vecs.append(i)
    #print(vecs)
    return np.array(vecs[:num_classes])

def generate_cSBM(d,lamb,mu,num_features,num_nodes,num_classes):
    c_in = d+lamb*np.sqrt(d) # c_in/c_out as described in the equations
    c_out = d-lamb*np.sqrt(d) 
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes
    
    adj, communities = generate_SSBM(num_nodes,num_classes,p_in,p_out) # obtains the graph structure
    u = np.random.normal(0,1/num_features,(num_features)) # obtains the random normal vector u
    Z = np.random.normal(0,.2,(num_nodes,num_features)) # obtains the random noise vector i presume
    v = communities # puts the groups into a format for the equations
    
    perms = generate_perms(num_classes,num_features)
    #print(communities)
    #print(perms)
    b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        b[i] = np.sqrt(mu/num_nodes)*(np.diag(perms[v[i]])@u) + Z[i]/np.sqrt(num_features)
    return adj,b,communities

from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import copy
class GCN(torch.nn.Module):# this is the torch geometric implementation of our GCN model like before, it
    # is a lot simpler to implement and way customizeable
    def __init__(self, in_feat, hid_feat, out_feat):
        super().__init__()
        self.conv1 = GCNConv(in_feat, hid_feat)
        self.conv2 = GCNConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x,dim=1)

def adj_to_list(a):
    adjList =[[],[]]
    for i in range(len(a)):
        for j in range(len(a[i])):
            if a[i][j]== 1:
                adjList[0].append(i)
                adjList[1].append(j)
    return adjList

def accuracy(preds,mask):# obtains the accuracy of the model
    correct = (preds[mask] == labels[mask]).sum()
    acc = int(correct)/int(mask.sum())
    return acc

d=10 # this is the average degree
lamb = 30 # difference in edge_densities, 0 indicates only node features are informative lamb>0 means more intra edges vs inter edges(homophily)
# lamb < 0 means less intra edges vs inter edges(heterophily)
mu = 50# difference between the means of the two classes, increasing this means increasing difference between class features
num_nodes = 200
num_features = 3
num_classes=7


# our hyperparameter for our hidden model
hidden_layers = 2
lr = .01
epochs = 400

#this sets up our variables to use in training
adj, b, labels = generate_cSBM(d,lamb,mu,num_features,num_nodes,num_classes)
edge_list = adj_to_list(adj)

# sets up model/optimizer
model = GCN(num_features,hidden_layers,num_classes)
optimizer = torch.optim.Adam(params=model.parameters(),lr = lr)

# creates some masks so we have stuff for training and validation
train_mask = torch.tensor([False,True]).repeat(num_nodes//2)
val_mask = ~train_mask

# turns all of our tensors into the desired format
edge_list = torch.Tensor(edge_list).to(torch.long)
b = torch.Tensor(b)
labels = torch.Tensor(labels).to(torch.long)

models = [0,1,2,3,4,5,6,7,8,9]
find_models= True
models_found = 0
all_accs = []
sub_accs = []
while find_models:
    adj, features, labels = generate_cSBM(d,lamb,mu,num_features,num_nodes,num_classes)
    edge_list = adj_to_list(adj)

    # sets up model/optimizer
    model = GCN(num_features,hidden_layers,num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(),lr = lr)

    # creates some masks so we have stuff for training and validation
    train_mask = torch.tensor([False,True]).repeat(num_nodes//2)
    val_mask = ~train_mask

    # turns all of our tensors into the desired format
    edge_list = torch.Tensor(edge_list).to(torch.long)
    features = torch.Tensor(features)
    labels = torch.Tensor(labels).to(torch.long)    

    model.train()# tells our model we are about to train
    for epoch in range(epochs):# runs through all the data 200 times
        optimizer.zero_grad()
        out = model(features,edge_list)

        train_loss = F.nll_loss(out[train_mask], labels[train_mask])        
        train_loss.backward()
        optimizer.step()

    model.eval()
    out = model(features,edge_list)
    test_acc = accuracy(out.max(1)[1],torch.ones(num_nodes).to(torch.bool))
    for i in range(10):
        if type(models[i]) == int and i != 0 and i != 2:
            if test_acc > .1*i and test_acc < .1*(i+1):
                models[i] = copy.deepcopy(model)
                print(test_acc)
                models_found += 1
                sub_accs.append(test_acc)
    if(models_found >= 8):
        find_models = False
        all_accs.append(sub_accs)
models = [models[1]] + models[3:]
models = np.array(models)
transfer_epochs = 10
for i in range(200):
    adj, features, labels = generate_cSBM(d,lamb,mu,num_features,num_nodes,num_classes)
    edge_list = adj_to_list(adj)

    # sets up model/optimizer
    model = GCN(num_features,hidden_layers,num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(),lr = lr)

    # creates some masks so we have stuff for training and validation
    train_mask = torch.tensor([False,True]).repeat(num_nodes//2)
    val_mask = ~train_mask

    # turns all of our tensors into the desired format
    edge_list = torch.Tensor(edge_list).to(torch.long)
    features = torch.Tensor(features)
    labels = torch.Tensor(labels).to(torch.long)
    sub_accs = []
    for imodel in models:
        model = copy.deepcopy(imodel)
        model.train()# tells our model we are about to train
        for epoch in range(transfer_epochs):# runs through all the data 200 times
            optimizer.zero_grad()
            out = model(features,edge_list)

            train_loss = F.nll_loss(out[train_mask], labels[train_mask])        
            train_loss.backward()
            optimizer.step()

        model.eval()
        out = model(features,edge_list)
        test_acc = accuracy(out.max(1)[1],torch.ones(num_nodes).to(torch.bool))
        sub_accs.append(test_acc)
    all_accs.append(sub_accs)
    print(f"time : {i} sub_accuracies: {sub_accs}")

    np.savetxt("sbm_gnn_transfer",np.array(all_accs))


