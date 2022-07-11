from torch_geometric.nn import GCNConv, GATConv,SAGEConv
from torch_geometric.nn import DenseGCNConv
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import copy
import torch

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

class NN(torch.nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat):
        super().__init__()
        self.lin1 = nn.Linear(in_feat, hid_feat)
        self.lin2 = nn.Linear(hid_feat, out_feat)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.lin1(x))
        x = F.dropout(x, training= self.training)
        x = self.lin2(x)
        return F.log_softmax(x,dim=1)

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