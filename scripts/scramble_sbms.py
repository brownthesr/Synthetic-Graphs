"""This is the file where we scramble the edge and feature data for 
the real world datasets.
"""
from torch_geometric.datasets import CitationFull,Amazon,Flickr,Yelp,IMDB,GitHub,FacebookPagePage,LastFMAsia,DeezerEurope,PolBlogs
import pandas as pd
import networkx as nx
from torch_geometric.nn import GCNConv,SAGEConv
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
import torch
class GCN(torch.nn.Module):# this is the torch geometric implementation of our GCN model like before, it
    # is a lot simpler to implement and way customizeable
    def __init__(self, in_feat, hid_feat, out_feat):
        super().__init__()
        self.conv1 = SAGEConv(in_feat, hid_feat)
        self.conv2 = SAGEConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x,dim=1)
def accuracy(preds,mask,data):# obtains the accuracy of the model
    correct = (preds[mask] == data.y[mask]).sum()
    acc = int(correct)/int(mask.sum())
    return acc
def find_duplicate_edges(edges):
    dic = {}
    duplicates = 0
    for (a,b) in edges:
        if not (a,b) in dic:
            dic[(a,b)] = 1
        else:
            duplicates += 1
    return duplicates
def assess(read = True,model=None,optimizer = None,data = None,twice = False):
    model.train()# tells our model we are about to train
    for epoch in range(200+200*twice):# runs through all the data 200 times
        optimizer.zero_grad()
        out = model(data.x,data.edge_index)

        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        train_acc = accuracy(out.max(1)[1], data.train_mask,data)

        val_loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out.max(1)[1], data.val_mask,data)

        if(epoch %5 == 0) and read:
            print("epoch: {} training_loss: {} training_acc: {} val_loss: {} val_acc: {}".format(epoch,train_loss,train_acc,val_loss, val_acc))
            
        train_loss.backward()
        optimizer.step()
    model.eval()
    preds = model(data.x,data.edge_index).max(1)[1]
    acc = accuracy(preds,data.test_mask,data)
    print(acc)
    return acc

def scramble_edges(edge_list, labels, ids, num_classes,dataset):
    """scrambles edges randomly through the network

    Args:
        edge_list (Tensor): edge list
        labels (Tensor): labels
        ids (Tensor): the ids of each node
        num_classes (int): the number of nodes

    Returns:
        Tensor: new edge list
    """
    data = dataset[0]
    edge_df = pd.DataFrame(edge_list.numpy().T)
    np_edge_data = edge_list.numpy().T
    labels = labels.numpy()
    ids = ids.numpy()
    for start_group in range(dataset.num_classes):
        id_mask = labels == start_group
        start_ids = ids[id_mask]
        
        for end_group in range(start_group,dataset.num_classes):
            id_mask = labels == end_group
            end_ids = ids[id_mask]

            start_mask_1 = edge_df[0].isin(start_ids)
            end_mask_1 = edge_df[1].isin(end_ids)
            total_mask_1 = np.logical_and(start_mask_1,end_mask_1)
            start_mask_2 = edge_df[1].isin(start_ids)
            end_mask_2 = edge_df[0].isin(end_ids)
            total_mask_2 = np.logical_and(start_mask_2,end_mask_2)
            total_mask = np.logical_or(total_mask_1,total_mask_2)

            total_mask = total_mask.to_numpy()

            for i in range(2):
                start_points = np_edge_data[total_mask][:,0]
                end_points = np_edge_data[total_mask][:,1]
                new_ends = np.random.permutation(end_points)
                # print(n_edge_data[total_mask][:,1].shape)
                np_edge_data[total_mask] = np.vstack((start_points,new_ends)).T
    np_edge_data = np_edge_data.T
    self_edges = np_edge_data[0] == np_edge_data[1]
    hyper_edges = find_duplicate_edges(np_edge_data.T)

    print(f"Percent of self_edges is {self_edges.sum()/len(self_edges)} Hyper edges is {hyper_edges/len(self_edges)}")
    data.edge_index = torch.tensor(np_edge_data).long()
    return torch.Tensor(np_edge_data).long()
def scramble_feat(feats, labels, num_classes):
    """scrambles the edges

    Args:
        feats (Tensor): Tensor of features
        labels (Tensor): Tensor of labels
        num_classes (int): the number of classes

    Returns:
        Tensor: the scrambled feature data by gaussians
    """
    feats = feats.numpy()
    labels = labels.numpy()
    
    for current_class in range(num_classes):
        mask = labels == current_class
        current_feat = feats[mask]
        num_feats = len(current_feat[0])
        num_nodes = len(current_feat)
        mean_list = np.mean(current_feat, axis=0)
        std_list = np.std(current_feat,axis = 0)

        new_feats = np.random.normal(mean_list,std_list, (num_nodes,num_feats))
        feats[mask] = new_feats
    return torch.Tensor(feats)

def run_dataset(name,t):

    if name == "Cora" or name == "DBLP" or name == "Citeseer" or name == "Pubmed":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = CitationFull(root=f'/tmp/{name}', name=name,transform = transform)
    elif name == "Computers" or name == "Photo":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = Amazon(root=f'/tmp/{name}', name=name,transform = transform)
    elif name == "Flickr":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = Flickr(root=f'/tmp/{name}',transform = transform)
    elif name == "Yelp":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = Yelp(root=f'/tmp/{name}',transform = transform)
    elif name == "IMDB":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = IMDB(root=f'/tmp/{name}',transform = transform)
    elif name == "GitHub":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = GitHub(root=f'/tmp/{name}',transform = transform)
    elif name == "FacebookPagePage":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = FacebookPagePage(root=f'/tmp/{name}',transform = transform)
    elif name == "LastFMAsia":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = LastFMAsia(root=f'/tmp/{name}',transform = transform)
    elif name == "DeezerEurope":
        transform = T.Compose([T.RandomNodeSplit(split="random", num_train_per_class = 20, num_val=1000, num_test=1000),
            T.TargetIndegree(),])
        dataset = DeezerEurope(root=f'/tmp/{name}',transform = transform)




    data = dataset[0]
    model = GCN(data.num_features,32,dataset.num_classes)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = .01,weight_decay=5e-4)
    data = data.cuda()
    model = model.cuda()

    normal = assess(False,model,optimizer,data)
    data = data.cpu()
    np_edge_data = data.edge_index.T.numpy()
    np_edge_data = np_edge_data.T
    data.edge_index = torch.tensor(np_edge_data).long()

    if t == "both":
        data.x = scramble_feat(data.x,data.y,dataset.num_classes)
        data.edge_index = scramble_edges(data.edge_index,data.y,torch.arange(data.num_nodes),dataset.num_classes,dataset)
    elif t == "edges":
        data.edge_index = scramble_edges(data.edge_index,data.y,torch.arange(data.num_nodes),dataset.num_classes,dataset)
    elif t == "feats":
        data.x = scramble_feat(data.x,data.y,dataset.num_classes)



    model = GCN(data.num_features,32,dataset.num_classes)
    optimizer = torch.optim.Adam(params = model.parameters(), lr = .01,weight_decay=5e-4)
    data = data.cuda()
    model = model.cuda()
    scrambled = assess(False,model,optimizer,data)
    return normal,scrambled

#"Cora","Citeseer","Pubmed","DBLP","Computers","Photo","Flickr","GitHub","FacebookPagePage"
datasets = ["Cora","Citeseer","Pubmed","DBLP","Computers","Photo","Flickr","GitHub","FacebookPagePage","LastFMAsia","DeezerEurope"]
# plt.ion()
runs = 5
typ = ["edges","both","feats"]
for t in typ:
    li = []
    for i,a in enumerate(datasets):
        normal_avg = 0
        scrambled_avg = 0
        for j in range(runs):
            normal,scrambled = run_dataset(a,t)
            normal_avg += normal/runs
            scrambled_avg += scrambled/runs
        print(f"Finished with dataset {a} {normal_avg} {scrambled_avg}")
        li.append([(normal_avg),(scrambled_avg),i])
        np.savetxt(f"scramble_{t}.txt",np.array(li))