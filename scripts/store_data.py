"""
This file runs different models of GNNs on a wide variety of graph data
and stores that data in the data folder
"""
import sys
from copy import deepcopy
import numpy as np
import torch
import time as time
import torch.nn as nn
import torch.nn.functional as F
from src.generate_data import *
from src.models import *
from src.utils import *
from torch_geometric.data import Data
from src.Graph_transformer.models import *
from torch_geometric.loader import DataLoader
from src.Graph_transformer.data import *
from itertools import permutations
from sklearn.cluster import SpectralClustering
import igraph as ig
# import graph_tool.all as gt
import leidenalg as la

# these are to support Parallelizability
comp_id = int(sys.argv[1])
# The maximum number of computers we will be using
MAX_COMPS = 200.0

# Parameter ranges
# features are informative lamb>0 means more intra edges vs inter edges(homophily)
# lamb < 0 means less intra edges vs inter edges(heterophily)
MAX_MU = float(sys.argv[2])
MAX_LAMB = float(sys.argv[3])


d=float(sys.argv[4])# this is the average degree
num_nodes = int(sys.argv[5])
num_features = int(sys.argv[6])
num_classes=int(sys.argv[7])
degree_corrected = (sys.argv[8] == "True")
max_values = sys.argv[9] == "True"
device = "cpu"

mu = 0 + MAX_MU/MAX_COMPS * comp_id# difference between the means
lamb = -10 # difference in edge_densities, 0 indicates only node
# of the two classes, increasing this means increasing difference between class features
gamma = 2.5
# this is so we can better parralelize things


# our hyperparameter for our hidden model
hidden_layers = 16
lr = .01
epochs = 400


runs = 10
all_accs = []
models = [GCN]# this is where you add the models that you want to run
# even further

def mu_loop(mu, lamb, model_type):
    """This is the main loop where we loop over mu values

    Args:
        mu (double): The beginning mu value
        lamb (double): The beginning lambda value
        model_type (int): The model type
    """
    while mu < MAX_MU/MAX_COMPS*(comp_id+1)-.00001:
        lamb = -3
        lamb_loop(mu, lamb,model_type)
        mu += MAX_MU/200
        if models[model_type].string() == "Spectral" or models[model_type].string() == "Leiden":
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
    average_accs = []
    if models[model_type].string() == "Spectral":
        for run in range(runs):
            model = SpectralClustering(num_classes,affinity = "precomputed", n_init = 100)
            test_adj, test_mask, test_labels = get_dataset(mu,lamb, adj =True)
            model.fit(test_adj)
            out = model.labels_
            possible_labels = np.arange(num_classes)
            accuracies = []
            for curr_label in permutations(possible_labels):
                perm = np.array(curr_label)
                acc1 = accuracy(out,test_mask,perm[test_labels])
                accuracies.append(acc1)
            partial_acc = np.max(accuracies)
            average_accs.append(partial_acc)

        if max_values:
            return np.max(average_accs)
        else:
            return np.mean(average_accs)
    if models[model_type].string() == "GraphTool":
        for run in range(runs):
            g = gt.Graph(directed=False)
            train_edge_list, train_b, train_labels, train_mask, \
                test_edge_list, test_b, test_labels, test_mask = get_dataset(mu,lamb)
            test_edge_list = test_edge_list.numpy()
            test_labels = test_labels.numpy()
            test_mask = test_mask.numpy()
            g.add_vertex(num_nodes)
            g.add_edge_list(test_edge_list)
            state = gt.minimize_blockmodel_dl(g)
            communities = list(state.get_blocks())
            communities = np.array(communities)
            old_labels = list(set(communities))
            new_labels = np.arange(len(old_labels))
            simplifier = dict(zip(old_labels,new_labels))
            simplified_communities = np.vectorize(simplifier.get)(communities)

            accuracies = []
            for cur_label in permutations(new_labels):
                perm = np.array(curr_label)
                acc1 = accuracy(perm[simplified_communities],test_mask,test_labels)
                accuracies.append(acc1)
            partial_acc = np.max(accuracies)
            average_accs.append(partial_acc)
        if max_values:
            return np.max(average_accs)
        else:
            return np.mean(average_accs)
    if models[model_type].string() == "Leiden":
        for run in range(runs):
            model = Leiden()
            test_adj, test_mask, test_labels = get_dataset(mu,lamb, adj =True)
            g = ig.Graph.Adjacency((test_adj> 0).tolist())
            if lamb >= 0:
                partition = la.CPMVertexPartition(g, 
                                  initial_membership=np.random.choice(num_classes, num_nodes),
                                  resolution_parameter=0.5)
                opt = la.Optimiser()
                opt.consider_empty_community = False
                opt.optimise_partition(partition,n_iterations = 100)
                out = partition.membership
            else:
                p_01,p_0,p_1 = la.CPMVertexPartition.Bipartite(g, 
                                    types=np.random.choice(num_classes, num_nodes),
                                    resolution_parameter_01=0.5)
                opt = la.Optimiser()
                opt.consider_empty_community = False

                opt.optimise_partition_multiplex([p_01, p_0, p_1],
                                       layer_weights=[1, -1, -1])
                out = p_1.membership
            
            out = np.array(out)
            possible_labels = np.arange(num_classes)
            accuracies = []
            for curr_label in permutations(possible_labels):
                perm = np.array(curr_label)
                acc1 = accuracy(out,test_mask,perm[test_labels])
                accuracies.append(acc1)
            partial_acc = np.max(accuracies)
            average_accs.append(partial_acc)
        if max_values:
            return np.max(average_accs)
        else:
            return np.mean(average_accs)
    if models[model_type].string() == "Graph_transformer":
        for run in range(runs):
            model = GraphTransformer(in_size = num_features,
                                    num_class=num_classes,
                                    d_model = 128,
                                    dim_feedforward=2*128,
                                    num_layers=2,
                                    gnn_type='gcn',
                                    use_edge_attr=False,
                                    num_heads = 8,
                                    in_embed=False,
                                    se="gnn",
                                    use_global_pool=False
                                    )
            optimizer = torch.optim.Adam(params=model.parameters(),lr = lr)
            train_edge_list, train_b, train_labels, train_mask, \
                test_edge_list, test_b, test_labels, test_mask = get_dataset(mu,lamb)
            train_data = [Data(x=train_b,edge_index=train_edge_list.long(),y=train_labels)]
            test_data = [Data(x=test_b,edge_index=test_edge_list.long(),y=test_labels)]
            train_dset = GraphDataset(train_data)
            test_dset = GraphDataset(test_data)
            train_loader = DataLoader(train_dset,batch_size = 1,shuffle = False)
            test_loader = DataLoader(test_dset,batch_size = 1,shuffle = False)
            model.cuda()
            #optimizer.cuda()
            test_mask.cuda()
            model.train()
            for train_batch in train_loader:
                train_batch.cuda()
                train_transformer(train_batch,optimizer,model,torch.ones(1000).bool())
            
            for test_batch in test_loader:
                test_batch.cuda()
                partial_acc = transformer_acc(model,test_batch,torch.ones(1000).bool())
            average_accs.append(partial_acc)
        if max_values:
            return np.max(average_accs)
        else:
            return np.mean(average_accs)

    for run in range(runs):
        np.random.seed(int(time.time()))
        torch.manual_seed(int(time.time()))
        model,optimizer = get_model_data(model_type)
        train_edge_list, train_b, train_labels, train_mask, \
            test_edge_list, test_b, test_labels, test_mask = get_dataset(mu,lamb)
        train_edge_list, train_b, train_labels, train_mask, \
            test_edge_list, test_b, test_labels, test_mask = train_edge_list.to(device), train_b.to(device), train_labels.to(device), train_mask.to(device), \
            test_edge_list.to(device), test_b.to(device), test_labels.to(device), test_mask.to(device)
        model.train()
        train_model(model, optimizer, train_b, train_edge_list,train_mask,train_labels)
        partial_acc = get_acc(model, test_b, test_edge_list, test_mask, test_labels)
        average_accs.append(partial_acc)
    if max_values:
        return np.max(average_accs)
    else:
        return np.mean(average_accs)

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
        np.savetxt(f"{num_classes}_DC_{models[model_type].string()}({comp_id}).txt",all_accs)
    else:
        np.savetxt(f"{num_classes}_{models[model_type].string()}({comp_id}).txt",all_accs)

def get_model_data(model_type):
    """Obtains objects for the model

    Args:
        model_type (int): The model type

    Returns:
        Model: the model to run tests on
        optimizer: an optimizer to optimize the model
    """
    model = models[model_type](num_features,hidden_layers,num_classes).to(device)
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

def train_transformer(data,optimizer,model,mask):
    for epoch in range(epochs):# runs through all the data 200 times

        optimizer.zero_grad()
        out = model(data)
        #print(out.shape,data.y.shape)
        train_loss = F.cross_entropy(out, data.y.squeeze())
        train_loss.backward()
        optimizer.step()

def transformer_acc(model, data,test_mask):
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
    out = model(data)
    acc = accuracy(out.max(1)[1],test_mask,data.y.squeeze())
    return acc

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
    mu_loop(mu, lamb, i)


