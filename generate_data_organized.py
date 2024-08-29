import sys
from copy import deepcopy
import numpy as np
import torch
from statistics import mean
import torch.nn as nn
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from generate_data2 import *
from src.models import *
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from src.utils import *
from torch_geometric.data import Data
from src.Graph_transformer.models import *
from torch_geometric.loader import DataLoader
from src.Graph_transformer.data import *
from itertools import permutations
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
    folder = "runs"
    if args.degree_corrected:
        np.savetxt(f"{folder}/{num_classes}_DC_{model_type.string()}({args.comp_id}).txt",all_accs)
    else:
        np.savetxt(f"{folder}/{num_classes}_{model_type.string()}({args.comp_id}).txt",all_accs)
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
def train_GPS(model,optimizer,dataset):
    train_edge_list, train_features, train_labels = dataset
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    train_data = transform(Data(x=train_features,edge_index=train_edge_list.long(),y=train_labels))
    model.train()
    data=  train_data.to(args.device)
    for epoch in range(args.epochs):# runs through all the data 200 times

        optimizer.zero_grad()
        out = model(data.x,data.edge_index, data.pe)
        #print(out.shape,data.y.shape)
        train_loss = F.cross_entropy(out, data.y.squeeze())
        train_loss.backward()
        optimizer.step()
def get_data(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes):
    return get_csbm(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes),get_csbm(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes)

def get_csbm(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes):
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
    edge_list = torch.tensor(list(sbm_graph.edges()), dtype=torch.long).t()
    n = class_size * num_classes
    labels = torch.arange(num_classes).repeat_interleave(class_size)

    features = torch.randn((n,num_features))
    features[torch.arange(n),labels] += mu
    
    
    return edge_list, features, labels
    
#changing getcsbm to model trevor's theoretical work. run it to get results testing the theoretical results. changes it to be diametrically opposed
def get_csbm2(num_nodes, num_features, average_degree, mu, lamb, degree_corrected, num_classes):
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
    edge_list = torch.tensor(list(sbm_graph.edges()), dtype=torch.long).t()
    n = class_size * num_classes
    labels = torch.arange(num_classes).repeat_interleave(class_size)

    features = torch.randn((n,num_features))
    
    #differing the seperation parameter to one dimension and to be subtracted from the other gaussian cloud
    features[labels==0,0]+=mu
    features[labels==1,0]-=mu
    #commented out code is what was previously used
    # features[torch.arange(n),labels] += mu
    
    
    return edge_list, features, labels

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
parser.add_argument("--runs", type=int,default=10, help= "The number of runs to max over")
parser.add_argument("--epochs", type=int,default=400, help= "The number of epochs we are planning to run")
parser.add_argument("--device", type=str,default="cuda", help= "The number of epochs we are planning to run")
parser.add_argument('-models', nargs='+', help='The models you will be passing through', required=True)

args = parser.parse_args()
print(args)
mu_range = [args.comp_id/args.max_comps*args.mu_max]
lambda_range = torch.linspace(args.lambda_min,args.lambda_max,121)
models = [eval(m) for m in args.models]
print(models)
for model_type in models:
    all_accs = []   
    if model_type in [GCN, SAGE, GAT]:
        for mu in mu_range:
            for lamb in lambda_range:
                accs = []
                for run in range(args.runs):
                    train_dataset, test_dataset = get_data(num_nodes=args.num_nodes, num_features=args.num_features, 
                                                           average_degree=args.average_degree, mu=mu, lamb=lamb, 
                                                           degree_corrected=args.degree_corrected, num_classes=args.num_classes)
                    model, optimizer = get_model(model_type,args.num_features, args.hidden_features, args.num_classes)
                    model = model.to(args.device)
                    train_model(model,optimizer,train_dataset)
                    accs.append(accuracy_1(torch.argmax(model(test_dataset[1].to(args.device),test_dataset[0].to(args.device)),dim=-1),test_dataset[2].to(args.device)))
                print(max(accs),lamb,mu)
                store_acc(max(accs),lamb,mu,model_type,args.num_classes)
    elif model_type is  GraphTransformer:
        for mu in mu_range:
            for lamb in lambda_range:
                accs = []
                for run in range(args.runs):
                    train_dataset, test_dataset = get_data(args.num_nodes, args.num_features, args.average_degree, mu, lamb, args.degree_corrected, args.num_classes)
                    model = GraphTransformer(in_size = args.num_features,
                                    num_class=args.num_classes,
                                    d_model = args.hidden_features,
                                    dim_feedforward=2*args.hidden_features,
                                    num_layers=2,
                                    gnn_type='gcn',
                                    use_edge_attr=False,
                                    num_heads = 1,
                                    in_embed=False,
                                    se="gnn",
                                    use_global_pool=False
                                    ,device = args.device
                                    )
                    optimizer = torch.optim.Adam(params=model.parameters(),lr = args.lr)
                    train_transformer_model(model,optimizer,train_dataset)
                    test_edge_list, test_features, test_labels = test_dataset
                    test_data = [Data(x=test_features,edge_index=test_edge_list.long(),y=test_labels)]
                    test_dset = GraphDataset(test_data)
                    test_loader = DataLoader(test_dset,batch_size = 1,shuffle = False)
                    for batch in test_loader:
                        batch = batch.to(args.device)
                        out = model(batch)
                        accs.append(accuracy_1(torch.argmax(out,dim=-1),test_dataset[2].to(args.device)))
                print(max(accs),lamb,mu)
                store_acc(max(accs),lamb,mu,model_type,args.num_classes)
    elif model_type == GPS:
        for mu in mu_range:
            for lamb in lambda_range:
                accs = []
                for run in tqdm(range(args.runs)):
                    train_dataset, test_dataset = get_data(args.num_nodes, args.num_features, args.average_degree, mu, lamb, args.degree_corrected, args.num_classes)
                    model, optimizer = get_model(model_type,args.num_features, args.hidden_features, args.num_classes)
                    model = model.to(args.device)
                    train_GPS(model,optimizer,train_dataset)
                    test_edge_list, test_features, test_labels = test_dataset
                    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

                    test_data = transform(Data(x=test_features,edge_index=test_edge_list.long(),y=test_labels))
                    data = test_data.to(args.device)
                    out = model(data.x,data.edge_index,data.pe)
                    accs.append(accuracy_1(torch.argmax(out,dim=-1),test_dataset[2].to(args.device)))
                print(max(accs),lamb,mu)
                store_acc(max(accs),lamb,mu,model_type,args.num_classes)