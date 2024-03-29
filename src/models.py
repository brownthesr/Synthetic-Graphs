"""
Contains all of the models that we want to run over our data.
GCN/GAT/SAGE, NN, and GCNModified(specifying where to add
comvolutions)
"""
from torch_geometric.nn import GCNConv, GATConv,SAGEConv
from torch_geometric.nn import DenseGCNConv
import torch.nn as nn
from itertools import permutations 
import torch.nn.functional as F
import torch

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
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index):
        """
        Runs forward propagation
        """
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
        #x = self.activation(self.convh(x,edge_index))
        #x = F.dropout(x,training=self.training)
        x = self.conv2(x, edge_index)
        if self.log_soft is True:
            return F.log_softmax(x,dim=1)
        else:
            return x
    def string():
        return "GCN"

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

    def forward(self, x,edge_index):
        """
        Runs forward propagation
        """
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
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
        self.conv1 = GATConv(in_feat, hid_feat)
        #self.convh = GCNConv(hid_feat,hid_feat)
        self.conv2 = GATConv(hid_feat, out_feat)
        self.activation = nn.ReLU()
        self.log_soft = log_soft
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index):
        """
        Runs forward propagation
        """
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
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
        x = F.dropout(x, training= self.training)
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

        Adj is included just so that we can lop over this
        """
        x = self.activation(self.lin1(x))
        #x = F.dropout(x, training= self.training)
        x = self.lin2(x)
        return F.log_softmax(x,dim=1)
    def string():
        return "NN"

class GCNModified(nn.Module):
    """
    This is a GCN that only applies graph convolutions
    where specified
    """
    def __init__(self,in_feat,hid_feat,out_feat, l1_conv,l2_conv):
        super().__init__()
        self.conv1 = DenseGCNConv(in_feat,hid_feat)
        self.conv2 = DenseGCNConv(hid_feat,out_feat)

        self.l1_conv = l1_conv
        self.l2_conv = l2_conv
        self.activation = nn.ReLU()
        self.final_activation = nn.Sigmoid()
    def forward(self, feat, adj):
        """
        Performs one round of forward propagation
        """
        x = self.activation(self.conv1(feat,torch.linalg.matrix_power(adj,self.l1_conv)))
        x = nn.functional.dropout(x,.5)
        x = self.final_activation(self.conv2(x,torch.linalg.matrix_power(adj,self.l2_conv)))
        return x
