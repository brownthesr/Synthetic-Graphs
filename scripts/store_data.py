"""
This file runs different models of GNNs on a wide variety of graph data
and stores that data in the data folder
"""
import numpy as np
import torch
from scipy.stats import poisson
from src.generate_data import *
from src.models import *
from src.utils import *
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy



# these are the hyperparameters to generate the graph/features, still unsure how tweaking them would change the graph
d=10 # this is the average degree
lamb = 0 # difference in edge_densities, 0 indicates only node features are informative lamb>0 means more intra edges vs inter edges(homophily)
# lamb < 0 means less intra edges vs inter edges(heterophily)
#lamb = 3.15
import sys
comp_id = int(sys.argv[1])
max_comps = 1.0
mu = 0 + 6.0/max_comps * comp_id# difference between the means of the two classes, increasing this means increasing difference between class features
a = 0
num_nodes = 1000
num_features = 10
num_classes=2


# our hyperparameter for our hidden model
hidden_layers = 10
lr = .01
epochs = 400


runs = 120
all_accs = []
while mu < 6/max_comps*(comp_id+1)-.01:# repeat this untill our mu reaches 0
    #for lamb_i in range(runs):
    avg = 0
    lamb_i = 0# this just tells us if we can start tracking averages
    lamb = 0# we need to reset lambda every loop
    while lamb <= 1:# repeat this loop until our value of lambda can solve exact recovery
        if lamb_i  > 5:
            computing_accs = np.array(all_accs)
            avg = computing_accs[-5:,0].mean()
        lamb_i += 1
        train_adj, train_b, train_labels,test_adj,test_b,test_labels = generate_cSBM_modified(d=d,lamb=lamb,num_classes=num_classes,num_features=num_features,o_distance=mu,num_nodes=num_nodes)#generate_DC_SBM(num_nodes,num_classes,num_features,lamb,mu)

        # sets up model/optimizer
        model = NN(num_features,hidden_layers,num_classes)
        model
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
            out = model(train_b)
            train_loss = F.nll_loss(out[train_mask], train_labels[train_mask])
            train_loss.backward()
            optimizer.step()
        model.eval()
        out = model(test_b)
        test_acc = accuracy(out.max(1)[1],test_mask,test_labels)
        all_accs.append([test_acc,lamb,mu])
        print("test_acc: ", test_acc,"lambda:",lamb,"mu:",mu)
        np.savetxt(f"data/DC_SBM_NN_test({comp_id}).txt",all_accs)
        lamb +=10
    mu +=.03

