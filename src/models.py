"""
Contains all of the models that we want to run over our data.
GCN/GAT/SAGE, NN, and GCNModified(specifying where to add
comvolutions)
"""
from torch_geometric.nn import GCNConv, GATv2Conv,SAGEConv, GPSConv,GINConv
from torch_geometric.nn import DenseGCNConv
import torch.nn as nn
import torch_geometric
from itertools import permutations 
import  matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

class b(torch_geometric.nn.MessagePassing):
    def forward(self,x,edge_index):
        return self.propagate(edge_index,x=x)

class GCN(torch.nn.Module):
    """
    Pytorch_Geometric implementation of GCN
    """
    def __init__(self, in_feat, hid_feat, out_feat, log_soft = True):
        """
        Constructor of class
        """
        super().__init__()
        self.conv1 = GCNConv(in_feat, hid_feat)
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = GCNConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        self.log_soft = log_soft
        self.propagate = b()
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index,pe=None):
        """
        Runs forward propagation
        """
        x = self.activation(self.conv1(x, edge_index))
        # x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        if self.log_soft is True:
            return F.log_softmax(x,dim=1)
        else:
            return x
    def string():
        return "GCN"

class GPS(torch.nn.Module):
    """
    Pytorch_Geometric implementation of GCN
    """
    def __init__(self, in_feat, hid_feat, out_feat, log_soft = True):
        """
        Constructor of class
        """
        super().__init__()
        self.nn1 = nn.Linear(in_feat,hid_feat//2)
        self.pe_norm = nn.BatchNorm1d(20)
        self.pe_lin = nn.Linear(20,hid_feat//2)
        self.conv1 = GPSConv(hid_feat, conv=GINConv(nn.Sequential(
                nn.Linear(hid_feat, hid_feat))
            ))
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = GPSConv(hid_feat, conv=GINConv(nn.Sequential(
                nn.Linear(hid_feat, hid_feat))
            ))
        self.activation = nn.ReLU()
        self.nn2 = nn.Linear(hid_feat,out_feat)
        self.log_soft = log_soft
        self.hid_feat = hid_feat
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index,pe):
        """
        Runs forward propagation
        """
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.nn1(x), self.pe_lin(x_pe)), 1)
        x = self.activation(self.conv1(x, edge_index))
        # x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        x = self.nn2(x)
        if self.log_soft is True:
            return F.log_softmax(x,dim=1)
        else:
            return x
    def string():
        return "GPS"

class SAGE(torch.nn.Module):
    """
    Pytorch_Geometric implementation of SAGE
    """
    def __init__(self, in_feat, hid_feat, out_feat, log_soft = True):
        """
        Constructor of class
        """
        super().__init__()
        self.conv1 = SAGEConv(in_feat, hid_feat)
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = SAGEConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        self.log_soft = log_soft
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index,pe=None):
        """
        Runs forward propagation
        """
        x = self.activation(self.conv1(x, edge_index))
        # x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        if self.log_soft is True:
            return F.log_softmax(x,dim=1)
        else:
            return x
    def string():
        return "SAGE"

class GAT(torch.nn.Module):
    """
    Pytorch_Geometric implementation of GAT
    """
    def __init__(self, in_feat, hid_feat, out_feat, log_soft = True):
        """
        Constructor of class
        """
        super().__init__()
        self.conv1 = GATv2Conv(in_feat, hid_feat)
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = GATv2Conv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        self.log_soft = log_soft
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index,pe=None):
        """
        Runs forward propagation
        """
        x = self.activation(self.conv1(x, edge_index))
        # x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        if self.log_soft is True:
            return F.log_softmax(x,dim=1)
        else:
            return x
    def string():
        return "GAT"

class Spectral:
    def string():
        return "Spectral"
class Leiden:
    def string():
        return "Leiden"
class GraphTool:
    def string():
        return "GraphTool"

class TopGCN(torch.nn.Module):
    """
    Pytorch_Geometric implementation of a Custom Topological GCN

    """
    def __init__(self, in_feat, hid_feat, out_feat):
        """
        Constructor of class
        """
        super().__init__()
        self.conv1 = nn.Linear(in_feat,hid_feat)#GCNConv(in_feat, hid_feat,add_self_loops=False)
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = GCNConv(hid_feat, out_feat,add_self_loops=False,normalize=False)
        self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index):
        """
        Runs forward propagation
        """
        x = self.activation(self.conv1(x))
        # x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        return x

class NN(torch.nn.Module):
    """
    Vanilla Neural Net implementation with torch
    """
    def __init__(self, in_feat, hid_feat, out_feat):
        """
        Constructor of class
        """
        super().__init__()
        self.lin1 = nn.Linear(in_feat, hid_feat)
        self.lin2 = nn.Linear(hid_feat, out_feat)
        self.activation = nn.ReLU()

    def forward(self, x,adj):
        """
        Runs forward propagation

        Adj is included just so that we can loop over this
        """
        x = self.activation(self.lin1(x))
        #x = F.dropout(x, training= self.training)
        x = self.lin2(x)
        return F.log_softmax(x,dim=1)
    def string():
        return "NN"

