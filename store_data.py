import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations,combinations,combinations_with_replacement

def generate_SSBM(n,c,p_intra,p_inter):# generates a symmetric SBM
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

def generate_perms(num_classes,num_feat):# generates where our point clouds will exist relative to the origin, uses a cube
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

def generate_perms_rand(num_classes,num_feat):# same thing, but initializes random orthogonal vectors instead
    vecs = []
    for i in range(num_classes):
        orthogonal = False
        while(not orthogonal):
            rand_vec = np.random.uniform(-10,10,num_feat)
            rand_vec = rand_vec/np.linalg.norm(rand_vec)
            orthogonal = True
            for i in range(len(vecs)):
                angle = np.arccos(rand_vec@vecs[i])
                if(angle < np.pi/2 - .1 or angle > np.pi/2 + .1):
                    orthogonal = False
        vecs.append(rand_vec)
        #print(len(vecs))
    return np.array(vecs)

def generate_cSBM(d,lamb,mu,num_features,num_nodes,num_classes):
    c_in = d+lamb*np.sqrt(d) # c_in/c_out as described in the equations
    c_out = d-lamb*np.sqrt(d) 
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes

    u = np.random.normal(0,1/num_features,(num_features)) # obtains the random normal vector u how far our clouds are from the origin

    train_adj, train_communities = generate_SSBM(num_nodes,num_classes,p_in,p_out) # obtains the graph structure
    train_Z = np.random.normal(0,.2,(num_nodes,num_features)) # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations
    
    perms = generate_perms_rand(num_classes,num_features)
    #print(communities)
    #print(perms)
    dist = np.sqrt(mu/num_nodes)
    train_b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@u) + train_Z[i]/np.sqrt(num_features)
        
    # recompute all this but for a test set
    test_adj, test_communities = generate_SSBM(num_nodes,num_classes,p_in,p_out)# change graph structure
    test_Z = np.random.normal(0,.2,(num_nodes,num_features)) # change the noise vector, but don't change the community centers
    test_b = np.zeros((num_nodes,num_features))
    test_v = test_communities
    for i in range(num_nodes):
        test_b[i] = dist*(np.diag(perms[test_v[i]])@u) + test_Z[i]/np.sqrt(num_features)
    
    return train_adj,train_b,train_communities, test_adj,test_b,test_communities

def generate_cSBM_modified(d,lamb,o_distance,num_features,num_nodes,num_classes):
    c_in = d+lamb*np.sqrt(d) # c_in/c_out as described in the equations
    c_out = d-lamb*np.sqrt(d) 
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes

    u = np.random.normal(0,1/num_features,(num_features)) # obtains the random normal vector u how far our clouds are from the origin

    train_adj, train_communities = generate_SSBM(num_nodes,num_classes,p_in,p_out) # obtains the graph structure
    train_Z = np.random.normal(0,.2,(num_nodes,num_features)) # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations
    
    perms = generate_perms_rand(num_classes,num_features)
    #print(communities)
    #print(perms)
    dist = o_distance
    train_b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@u) + train_Z[i]/np.sqrt(num_features)
        
    # recompute all this but for a test set
    test_adj, test_communities = generate_SSBM(num_nodes,num_classes,p_in,p_out)# change graph structure
    test_Z = np.random.normal(0,.2,(num_nodes,num_features)) # change the noise vector, but don't change the community centers
    test_b = np.zeros((num_nodes,num_features))
    test_v = test_communities
    for i in range(num_nodes):
        test_b[i] = dist*(np.diag(perms[test_v[i]])@u) + test_Z[i]/np.sqrt(num_features)
    
    return train_adj,train_b,train_communities, test_adj,test_b,test_communities
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
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = GCNConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
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

def accuracy(preds,mask,labels):# obtains the accuracy of the model
    correct = (preds[mask] == labels[mask]).sum()
    acc = int(correct)/int(mask.sum())
    return acc

# these are the hyperparameters to generate the graph/features, still unsure how tweaking them would change the graph
d=10 # this is the average degree
lamb = 0 # difference in edge_densities, 0 indicates only node features are informative lamb>0 means more intra edges vs inter edges(homophily)
# lamb < 0 means less intra edges vs inter edges(heterophily)
#lamb = 3.15
mu = 0# difference between the means of the two classes, increasing this means increasing difference between class features
a = 0
num_nodes = 1000
num_features = 80
num_classes=2


# our hyperparameter for our hidden model
hidden_layers = 10
lr = .01
epochs = 400


runs = 120
all_accs = []
for a in range(200):# repeat this untill our mu reaches 0
    #for lamb_i in range(runs):
    avg = 0
    lamb_i = 0# this just tells us if we can start tracking averages
    lamb = 0# we need to reset lambda every loop
    while lamb < 3:# repeat this loop until our value of lambda can solve exact recovery
        if lamb_i  > 5:
            computing_accs = np.array(all_accs)
            avg = computing_accs[-5:,0].mean()
        lamb_i += 1
        train_adj, train_b, train_labels,test_adj,test_b,test_labels = generate_cSBM_modified(d,lamb,mu,num_features,num_nodes,num_classes)
        

        # sets up model/optimizer
        model = GCN(num_features,hidden_layers,num_classes)
        model.cuda()
        optimizer = torch.optim.Adam(params=model.parameters(),lr = lr)

        # creates some masks so we have stuff for training and validation
        train_mask = torch.ones(num_nodes).bool().cuda()
        #train_mask = train_mask[np.random.permutation(num_nodes)]
        test_mask = torch.ones(num_nodes).bool().cuda()

        # turns all of our tensors into the desired format
        train_edge_list = adj_to_list(train_adj)
        train_edge_list = torch.Tensor(train_edge_list).to(torch.long).cuda()
        train_b = torch.Tensor(train_b).cuda()
        train_labels = torch.Tensor(train_labels).to(torch.long).cuda()

        test_edge_list = adj_to_list(test_adj)
        test_edge_list = torch.Tensor(test_edge_list).to(torch.long).cuda()
        test_b = torch.Tensor(test_b).cuda()
        test_labels = torch.Tensor(test_labels).to(torch.long).cuda()


        model.train()# tells our model we are about to train

        for epoch in range(epochs):# runs through all the data 200 times
            
            optimizer.zero_grad()
            out = model(train_b,train_edge_list)
            train_loss = F.nll_loss(out[train_mask], train_labels[train_mask])
            train_loss.backward()
            optimizer.step()
        model.eval()
        out = model(test_b,test_edge_list)
        test_acc = accuracy(out.max(1)[1],test_mask,test_labels)
        all_accs.append([test_acc,lamb,mu])
        print("test_acc: ", test_acc,"lambda:",lamb,"mu:",mu)
        np.savetxt("change_lambda_sublinearscale.txt",all_accs)
        lamb +=.05
    mu = .03*a

