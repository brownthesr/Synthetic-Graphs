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
MAX_MU = 2.0
MAX_LAMB = 3.0
mu = 0 + MAX_MU/MAX_COMPS * comp_id# difference between the means
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


runs = 20
all_accs = []
models = [Spectral]
degree_corrected = True

def mu_loop(mu, lamb, model_type):
    """This is the main loop where we loop over mu values

    Args:
        mu (double): The beginning mu value
        lamb (double): The beginning lambda value
        model_type (int): The model type
    """
    while mu < MAX_MU/MAX_COMPS*(comp_id+1)-.00001:
        lamb = 0
        lamb_loop(mu, lamb,model_type)
        mu += MAX_MU/200
        if models[model_type].string() == "Spectral":
            mu += 100

def lamb_loop(mu, lamb, model_type):
    """The main lambda loop

    Args:
        mu (double): Current mu value
        lamb (double): Starting lambda value
        model_type (int): model type id
    """
    while lamb <= MAX_LAMB:
        test_acc = runs_loop(mu, lamb, model_type)
        write_data(mu, lamb, test_acc, model_type)
        print("test_acc: ", test_acc,"lambda:",lamb,"mu:",mu)
        lamb += MAX_LAMB/60
        if models[model_type].string() == "NN":
            lamb += 100

def runs_loop(mu, lamb, model_type):
    """Averages over many runs on the same mu and lambda

    Args:
        mu (double): Current mu value
        lamb (double): Current lambda value
        model_type (int): model type id

    Returns:
        average_acc: the average accuracy
    """
    average_acc = 0
    if models[model_type].string() == "Spectral":
        for run in range(runs):
            model = SpectralClustering(num_classes,affinity = "precomputed", n_init = 100)
            test_adj, test_mask, test_labels = get_dataset(mu,lamb, adj =True)
            model.fit(test_adj)
            out1 = model.labels_
            out2 = 1-model.labels_ 
            acc1 = accuracy(out1,test_mask,test_labels)
            acc2 = accuracy(out2,test_mask,test_labels)
            partial_acc = np.max([acc1,acc2])
            average_acc += partial_acc/runs
        return average_acc
    for run in range(runs):
        model,optimizer = get_model_data(model_type)
        train_edge_list, train_b, train_labels, train_mask, \
            test_edge_list, test_b, test_labels, test_mask = get_dataset(mu,lamb)
        
        model.train()
        train_model(model, optimizer, train_b, train_edge_list,train_mask,train_labels)
        partial_acc = get_acc(model, test_b, test_edge_list, test_mask, test_labels)
        average_acc += partial_acc/runs
    return average_acc

def write_data(mu, lamb, test_acc, model_type):
    """Function in charge of writing data to files

    Args:
        mu (double): Current mu value
        lamb (double): Current lambda value
        test_acc (double): Average testing accuracy
        model_type (int): Model type identity
    """
    all_accs.append([test_acc,lamb,mu])
    if degree_corrected:
        np.savetxt(f"DC_{models[model_type].string()}({comp_id}).txt",all_accs)
    else:
        np.savetxt(f"{models[model_type].string()}({comp_id}).txt",all_accs)

def get_model_data(model_type):
    """Obtains objects for the model

    Args:
        model_type (int): The model type

    Returns:
        Model: the model to run tests on
        optimizer: an optimizer to optimize the model
    """
    model = models[model_type](num_features,hidden_layers,num_classes)
    optimizer = torch.optim.Adam(params=model.parameters(),lr = lr)
    return model, optimizer

def get_dataset(mu, lamb,adj = False):
    """Obtains the SBM dataset for the model

    Args:
        mu (double): Current mu value
        lamb (double): Current lambda value

    Returns:
        The information on the training and test sets
    """
    if degree_corrected:
        train_adj, train_b, train_labels, test_adj, test_b, test_labels= generate_cdcbm(
            avg_degree=d,degree_separation=lamb,num_classes=num_classes,num_features=num_features,
            origin_distance=mu,num_nodes=num_nodes,gamma=gamma)
    else:
        train_adj, train_b, train_labels, test_adj, test_b, test_labels= generate_csbm_modified(
            avg_degree=d,degree_separation=lamb,num_classes=num_classes,num_features=num_features,
            origin_distance=mu,num_nodes=num_nodes)
            # generate_DC_SBM(num_nodes,num_classes,num_features,lamb,mu)
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
    if not adj:
        return train_edge_list, train_b, train_labels, train_mask, \
            test_edge_list, test_b, test_labels, test_mask
    else:
        return test_adj, test_mask.numpy(), test_labels.numpy()

def train_model(model, optimizer, train_b, train_edge_list,train_mask,train_labels):
    """Trains the model

    Args:
        model (Model): Model to run
        optimizer (Optimizer): Optimizer to optimize model
        train_b (Tensor): Feature Data
        train_edge_list (Tensor): Edge list
        train_mask (Tensor): Masking on dataset
        train_labels (Tensor): Labels for dataset
    """
    model.train()# tells our model we are about to train

    for epoch in range(epochs):# runs through all the data 200 times

        optimizer.zero_grad()
        out = model(train_b,train_edge_list)
        train_loss = F.nll_loss(out[train_mask], train_labels[train_mask])
        train_loss.backward()
        optimizer.step()

def get_acc(model, test_b, test_edge_list, test_mask, test_labels):
    """Tests accuracy for model

    Args:
        model (Model): The model to test
        test_b (Tensor): The feature data
        test_edge_list (Tensor): The edge data
        test_mask (Tensor): The mask for our data
        test_labels (Tensor): The labels on our data points

    Returns:
        double: The total accuracy of the model
    """
    model.eval()
    out = model(test_b,test_edge_list)
    acc = accuracy(out.max(1)[1],test_mask,test_labels)
    return acc

for i in range(len(models)):
    all_accs = []
    mu_loop(mu, lamb, i)
