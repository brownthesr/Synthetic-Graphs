import sys
from copy import deepcopy
import numpy as np
import torch
from statistics import mean
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
# from generate_data2 import *
from src.models import *
from src.utils import *
from torch_geometric.data import Data
from src.Graph_transformer.models import *
from torch_geometric.loader import DataLoader
from src.Graph_transformer.data import *
from itertools import permutations
from scipy.stats import powerlaw
from sklearn.cluster import SpectralClustering

def accuracy_1(preds,labels):
    """
    Obtains the accuracy of a model
    """
    correct = (preds == labels).sum()
    acc = int(correct)/len(labels)
    return acc
def store_acc(test_acc, lamb, mu, model_type, num_classes):
    """Function in charge of writing data to files

    Args:
        mu (double): Current mu value
        lamb (double): Current lambda value
        test_acc (double): Average testing accuracy
        model_type (int): Model type identity
    """
    all_accs.append([test_acc,lamb,mu])
    folder = "maxes"
    if args.degree_corrected:
        np.savetxt(f"{folder}/{num_classes}_DC_{model_type.string()}.txt",all_accs)
    else:
        np.savetxt(f"{folder}/{num_classes}_{model_type.string()}.txt",all_accs)
def get_model(model_type, num_features, hidden_layers, out_features):
    """Obtains objects for the model

    Args:
        model_type (nn.Module): The model type
        num_features (int): The number of input features
        hidden_layers (int): The number of hidden layers
        out_features (int): The number of output features

    Returns:
        Model: the model to run tests on
        optimizer: an optimizer to optimize the model
    """
    model = model_type(num_features, hidden_layers, out_features)
    optimizer = torch.optim.Adam(params=model.parameters(),lr = args.lr)
    return model, optimizer

def train_model(model, optimizer, dataset):
    """Takes the max over many runs on the same mu and lambda

    Args:
        model (nn.Module): model to train
        optimizer (torch.optim): optimizer for model
        dataset (tuple): model type id
    """
    model = model.to(args.device)
    train_edge_list, train_features, train_labels = dataset
    train_edge_list,train_features, train_labels = train_edge_list.to(args.device),train_features.to(args.device), train_labels.to(args.device)

    model.train()

    for epoch in range(args.epochs):

        optimizer.zero_grad()
        out = model(train_features,train_edge_list)
        train_loss = F.nll_loss(out, train_labels)
        train_loss.backward()
        optimizer.step()
    torch.save(model.state_dict(),f"trained_models/model_with{mu}mu_{lamb}lamb")

def train_transformer_model(model,optimizer,dataset):
    train_edge_list, train_features, train_labels = dataset
    train_data = [Data(x=train_features,edge_index=train_edge_list.long(),y=train_labels)]
    train_dset = GraphDataset(train_data)
    train_loader = DataLoader(train_dset,batch_size = 1,shuffle = False)
    model.to(args.device)
    model.train()
    for data in train_loader:
        data=  data.to(args.device)
        for epoch in range(args.epochs):# runs through all the data 200 times

            optimizer.zero_grad()
            out = model(data)
            #print(out.shape,data.y.shape)
            train_loss = F.cross_entropy(out, data.y.squeeze())
            train_loss.backward()
            optimizer.step()
    torch.save(model.state_dict(),f"trained_models/model_with{mu}mu_{lamb}lamb")

def get_data(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes,ret_adj=False):
    return get_csbm(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes,ret_adj=ret_adj),get_csbm(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes,ret_adj=ret_adj)

def get_csbm(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes, ret_adj = False):
    c_in = average_degree+lamb*np.sqrt(average_degree) # c_in/c_out as described in the equations
    c_out = average_degree-lamb*np.sqrt(average_degree)
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes


    class_size = num_nodes // num_classes
    block_sizes = [class_size] * num_classes
    # Generate block probabilities
    p_matrix = np.full((num_classes, num_classes), p_out)
    np.fill_diagonal(p_matrix, p_in)
    # Generate SBM graph
    sbm_graph = nx.stochastic_block_model(block_sizes, p_matrix).to_directed()
    if degree_corrected:
        #Kaden added line 122-123
        for node in sbm_graph.nodes():
            sbm_graph.nodes[node]['degree']=1

        for i in range(num_classes):
            degrees = powerlaw.rvs(2, size=class_size).astype(int)
            for j, node in enumerate(range(i * class_size, (i + 1) * class_size)):
                sbm_graph.nodes[node]['degree'] = degrees[j]

            # Rewire edges based on degrees
            for u, v in list(sbm_graph.edges()):
                u_degree = sbm_graph.nodes[u]['degree']
                v_degree = sbm_graph.nodes[v]['degree']
                p_rewire = u_degree * v_degree / (num_nodes * average_degree)
                if np.random.rand() < p_rewire:
                    sbm_graph.remove_edge(u, v)
                    new_v = np.random.choice(list(sbm_graph.nodes()))
                    sbm_graph.add_edge(u, new_v)
    edge_list = torch.tensor(list(sbm_graph.edges()), dtype=torch.long).t()
    n = class_size * num_classes
    labels = torch.arange(num_classes).repeat_interleave(class_size)

    features = torch.randn((n,num_features))
    features[torch.arange(n),labels] += mu
    
    if not ret_adj:
        return edge_list, features, labels
    else:
        return nx.to_numpy_array(sbm_graph),features.numpy(),labels.numpy()
    
parser = argparse.ArgumentParser()
parser.add_argument("--comp_id", type=int,default=0, help= "This is the index of the parralelization, basically it tells us what mu to run")
parser.add_argument("--max_comps", type=int,default=200, help= "The number of computers we are using in parrallel")
parser.add_argument("--mu_max", type=float,default=2.0, help= "We will use 200 different mu values between (0,max_mu)")
parser.add_argument("--lambda_min", type=float,default=-3.0, help= "We will use 121 different lambda values between (lambda_min,lambda_max)")
parser.add_argument("--lambda_max", type=float,default=3.0, help= "We will use 121 different lambda values between (lambda_min,lambda_max)")
parser.add_argument("--num_features", type=int,default=10, help= "This is the size of the input dimension")
parser.add_argument("--num_nodes", type=int,default=1000, help= "This is the size of the graph")
parser.add_argument("--num_classes", type=int,default=2, help= "This is the number of classes")
parser.add_argument("--average_degree", type=int,default=10, help= "This is the average degree of the graph")
parser.add_argument("--degree_corrected", type=bool,default=False, help= "This is whether or not we are calculating degree corrected graphs")
parser.add_argument("--hidden_features", type=int,default=16, help= "This is whether or not we are calculating degree corrected graphs")
parser.add_argument("--lr", type=float,default=.01, help= "The default learning rate")
parser.add_argument("--runs", type=int,default=1, help= "The number of runs to max over")
parser.add_argument("--epochs", type=int,default=400, help= "The number of epochs we are planning to run")
parser.add_argument("--device", type=str,default="cuda", help= "The number of epochs we are planning to run")
parser.add_argument('--models', nargs='+', help='The models you will be passing through')

args = parser.parse_args()
print(args)
mu_range = [args.comp_id/args.max_comps*args.mu_max]
lambda_range = torch.linspace(args.lambda_min,args.lambda_max,121)
models = [eval(m) for m in args.models]
print(models)
print("test")
for model_type in models:
    print("starting")
    all_accs = []  
    print(f'model type is{model_type}') 
    # if model_type is NN:
    if model_type is GCN or GAT:
        for mu in mu_range:
            for lamb in lambda_range:
                accs = []
                for run in range(args.runs):
                    train_dataset, test_dataset = get_data(num_nodes=args.num_nodes, num_features=args.num_features, 
                                                            average_degree=args.average_degree, mu=mu, lamb=0, 
                                                            degree_corrected=args.degree_corrected, num_classes=args.num_classes)
                    model, optimizer = get_model(model_type,args.num_features, args.hidden_features, args.num_classes)
                    model = model.to(args.device)
                    train_model(model,optimizer,train_dataset)
                    print("trained!")
                    accs.append(accuracy_1(torch.argmax(model(test_dataset[1].to(args.device),test_dataset[0].to(args.device)),dim=-1),test_dataset[2].to(args.device)))
                print(max(accs),lamb,mu)
                store_acc(max(accs),0,mu,model_type,args.num_classes)
    if model_type is Spectral:
        for lamb in lambda_range:
            accs = []
            for run in range(args.runs):
                model = SpectralClustering(args.num_classes,affinity = "precomputed", n_init = 100)
                train_dataset, test_dataset = get_data(num_nodes=args.num_nodes, num_features=args.num_features, 
                                                        average_degree=args.average_degree, mu=0, lamb=0, 
                                                        degree_corrected=args.degree_corrected, num_classes=args.num_classes,ret_adj=True)
                test_adj, test_mask, test_labels = test_dataset
                model.fit(test_adj)
                out = model.labels_
                possible_labels = np.arange(args.num_classes)
                accuracies = []
                for curr_label in permutations(possible_labels):
                    perm = np.array(curr_label)
                    acc1 = accuracy_1(out,perm[test_labels])
                    accuracies.append(acc1)
                partial_acc = np.max(accuracies)
                accs.append(partial_acc)
        