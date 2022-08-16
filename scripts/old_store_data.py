"""
This file runs different models of GNNs on a wide variety of graph data
and stores that data in the data folder
"""
import sys
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.generate_data import *
from src.models import *
from src.utils import *
from sklearn.cluster import SpectralClustering




# these are to support Parallelizability
comp_id = int(sys.argv[1])
MAX_COMPS = 1.0
mu = 0 + 6.0/MAX_COMPS * comp_id# difference between the means
# of the two classes, increasing this means increasing difference between class features

d=10 # this is the average degree
lamb = 0 # difference in edge_densities, 0 indicates only node
# features are informative lamb>0 means more intra edges vs inter edges(homophily)
# lamb < 0 means less intra edges vs inter edges(heterophily)
num_nodes = 1000
num_features = 10
gamma = 2.5
num_classes=2


# our hyperparameter for our hidden model
hidden_layers = 10
lr = .01
epochs = 400


runs = 10
all_accs = []

while mu < 6/MAX_COMPS*(comp_id+1)-.01:# repeat this untill our mu reaches 0
    #for lamb_i in range(runs):

    lamb = 0# we need to reset lambda every loop
    while lamb <= 3:# repeat this loop until our value of lambda can solve exact recovery
        
        train_adj, train_b, train_labels, test_adj, test_b, test_labels= generate_cdcbm(
            avg_degree=d,degree_separation=lamb,num_classes=num_classes,num_features=num_features,
            origin_distance=mu,num_nodes=num_nodes,gamma=gamma)
            # generate_DC_SBM(num_nodes,num_classes,num_features,lamb,mu)

        # sets up model/optimizer
        model = GCN(num_features,hidden_layers,num_classes)
        optimizer = torch.optim.Adam(params=model.parameters(),lr = lr)

        # creates some masks so we have stuff for training and validation
        train_mask = torch.ones(num_nodes).bool()
        #train_mask = train_mask[np.random.permutation(num_nodes)]
        test_mask = torch.ones(num_nodes).bool()

        # turns all of our tensors into the desired format
        train_edge_list = adj_to_list(train_adj)
        train_edge_list = torch.Tensor(train_edge_list).to(torch.long)
        train_b = torch.Tensor(train_b)
        train_labels = torch.Tensor(train_labels).to(torch.long)

        test_edge_list = adj_to_list(test_adj)
        test_edge_list = torch.Tensor(test_edge_list).to(torch.long)
        test_b = torch.Tensor(test_b)
        test_labels = torch.Tensor(test_labels).to(torch.long)


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
        np.savetxt(f"DC_mu_lambda_variation_GCN.txt",all_accs)
        lamb +=.05
    mu +=.03
