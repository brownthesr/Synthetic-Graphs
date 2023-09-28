"""This is the file where we scramble the edge and feature data for 
the real world datasets.
"""
from tqdm import tqdm
from torch_geometric.datasets import CitationFull,Amazon,Flickr,Yelp,IMDB,GitHub,FacebookPagePage,LastFMAsia,DeezerEurope,PolBlogs
import pandas as pd
from torch_geometric.loader import DataLoader
from src.Graph_transformer.data import *
import networkx as nx
from torch_geometric.nn import GCNConv,SAGEConv,GATConv
from matplotlib import pyplot as plt
from src.Graph_transformer.models import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
import torch
class GNN(torch.nn.Module):
    """Implementation of various Graph Neural Networks"""
    def __init__(self, in_feat, hid_feat, out_feat,type="GCN"):
        super().__init__()
        layer = None
        if type == "GCN":
            layer = GCNConv
        elif type == "GAT":
            layer = GATConv
        elif type == "SAGE":
            layer = SAGEConv
        self.conv1 = layer(in_feat, hid_feat)
        self.conv2 = layer(hid_feat, out_feat)
        self.activation = nn.ReLU()
        #self.dropout = nn.Dropout(p=.4)

    def forward(self, x,edge_index):
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, training= self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x,dim=1)

def accuracy(preds,mask,data):
    """Predicts the accuracy of our model"""
    correct = (preds[mask] == data.y[mask]).sum()
    acc = int(correct)/int(mask.sum())
    return acc

def find_duplicate_edges(edges):
    """Finds duplicate edges if any"""
    dic = {}
    duplicates = 0
    for (a,b) in edges:
        if not (a,b) in dic:
            dic[(a,b)] = 1
        else:
            duplicates += 1
    return duplicates

def train_transformer(data,optimizer,model,mask):
    """Trains the transformer on our data"""
    for epoch in range(200):

        optimizer.zero_grad()
        out = model(data)
        #print(out.shape,data.y.shape)
        train_loss = F.cross_entropy(out[mask], data.y[mask].squeeze())
        train_loss.backward()
        optimizer.step()

def assess(read = True,model=None,optimizer = None,data = None,twice = False,transformer = False):
    """Performs both training and test loops for our models
    
        Parameters
        ----------
        read (bool): Whether we should print off training progress
        model (GNN): The model we are using
        optimizer (torch.nn.optim): The optimizer we are using
        data (Data): The graph data to input into our model
        twice (bool): Whether to train twice
        transformer (bool): Whether we are training a transformer
    """
    if not transformer:
        model.train()# tells our model we are about to train
        for epoch in range(200+200*twice):# runs through all the data 200 times
            optimizer.zero_grad()
            out = model(data.x,data.edge_index)
            data.y = data.y.reshape(-1)
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
        # print("testing accuracy:",acc)
        return acc
    else:
        model.train()# tells our model we are about to train
        
        train_dset = [Data(x=data.x,edge_index=data.edge_index.long(),y=data.y).cuda()]
        train_dset = GraphDataset(train_dset)
        train_loader = DataLoader(train_dset,batch_size = 1, shuffle = False)
        model.cuda()
        optimizer.zero_grad()
        data.train_mask.cuda()
        for train_batch in train_loader:
            train_batch.cuda()
            train_transformer(train_batch,optimizer,model,data.train_mask)
        

        model.eval()
        data.test_mask.cuda()
        for test_batch in train_loader:
            test_batch.cuda()
            print("1 iteration DELETE ME")
            preds = model(test_batch).max(1)[1]
            acc = accuracy(preds,data.test_mask,data)
        # print(acc)
        return acc
    
def hSBM(n_nodes_per_subclass, intra_subclass_prob, inter_subclass_prob, interclass_prob, 
         mu, noise_std_dev=0.05):
    """
    Generate hierarchical stochastic block model (hSBM).
    
    Parameters:
    - n_nodes_per_subclass: Number of nodes in each subclass
    - intra_subclass_prob: Probability of connecting nodes within the same subclass
    - inter_subclass_prob: Probability of connecting nodes from different subclasses but same main class
    - interclass_prob: Probability of connecting nodes from different main classes
    - main_class_means: Mean features for the two main classes
    - subclass_means: Mean features for the subclasses of the two main classes
    - noise_std_dev: Standard deviation for feature noise
    
    Returns:
    - features: Node features as a numpy array
    - labels: Main class labels as a numpy array
    - edge_index: Edge indices as a torch tensor
    """
    # Create an empty graph
    G = nx.Graph()
    def generate_subclass_features(main_class_mean, distance=0.2, n_subclasses=5):
        # Generate features for subclasses forming a pentagon around the main class
        subclass_features = []
        for i in range(n_subclasses):
            theta = 2 * np.pi * i / n_subclasses
            x = main_class_mean[0] + distance * np.cos(theta)
            y = main_class_mean[1] + distance * np.sin(theta)
            subclass_features.append([x, y])
        return subclass_features
    
    # Containers for features and labels
    features = []
    labels = []
    main_class_mean  = np.random.normal(0,1/10,(10))
    while np.linalg.norm(main_class_mean) ==0:
        main_class_mean = np.random.normal(0,1/10,(10))
    main_class_mean = mu*main_class_mean/np.linalg.norm(main_class_mean)

    main_class_means = [main_class_mean,-main_class_mean]
    subclass_means = generate_subclass_features(main_class_means)
    a = 0
    # Generate nodes and features for each subclass
    for main_class_id, main_class_mean in enumerate(main_class_means):
        for subclass_id, subclass_mean_ in enumerate(subclass_means):
            for _ in range(n_nodes_per_subclass):
                subclass_mean = subclass_mean_[main_class_id]
                a+=1
                # Generate noisy feature for the node
                feature = np.random.normal(loc=subclass_mean, scale=.2, size=10)
                features.append(feature)
                labels.append(main_class_id)
                node_id = len(features) - 1
                G.add_node(node_id, feature=feature, main_class=main_class_id, subclass=subclass_id)
    
    # Connect nodes based on the given probabilities
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            node_i = nodes[i]
            node_j = nodes[j]
            main_class_i = G.nodes[node_i]['main_class']
            main_class_j = G.nodes[node_j]['main_class']
            subclass_i = G.nodes[node_i]['subclass']
            subclass_j = G.nodes[node_j]['subclass']
            
            # Determine connection probability
            if main_class_i == main_class_j:
                if subclass_i == subclass_j:
                    prob = intra_subclass_prob
                else:
                    prob = inter_subclass_prob
            else:
                prob = interclass_prob
            
            if np.random.rand() < prob:
                G.add_edge(node_i, node_j)
    
    # Convert edge list to torch tensor format
    edge_index = torch.tensor(list(G.edges())).T
    
    return np.array(features), np.array(labels), edge_index.T

def generate_orthogonal_vecs(num_vecs,num_dim):
    """Generates Orthogonal vectors

    Uses a random generation process to generate num_vecs
    orthogonal vectors in num_dim dimensions.

    Args:
        num_vecs (int): The number of vectors we want to return.
        num_dim (int): The number of dimensions that we can use.

    Returns:
        vecs (list): A list of randomly orthogonal vectors
    """
    vecs = []
    assert num_vecs <= num_dim
    for i in range(num_vecs):
        orthogonal = False
        while not orthogonal:
            rand_vec = np.random.uniform(-10,10,num_dim)
            rand_vec = rand_vec/np.linalg.norm(rand_vec)
            orthogonal = True
            for j in range(len(vecs)):
                angle = np.arccos(rand_vec@vecs[j])
                if(angle < np.pi/2 - .1 or angle > np.pi/2 + .1):
                    orthogonal = False
        vecs.append(rand_vec)
        #print(len(vecs))
    return np.array(vecs)

def generate_ssbm(num_nodes,num_classes,p_intra,p_inter,community = None):
    """Generates a SSBM.

    This function generates a Symmetric Stochastic Block model. It does
    this by creating blocks for in and out of class probability. Then
    it draws from a uniform distribution and uses the p_intra and p_inter
    probabilities to assign edges between specific nodes

    Args:
        num_nodes (int): The number of nodes.
        num_classes (int): The number of classes.
        p_intra (float): The probability of having
            in class connections.
        p_inter (float): The probability of having
            edges between classes
        community (list): Optional, may specify how
            the nodes are divided into communities.
            Automatically assigns communities if none
            are provided.

    Returns:
        Graph (list): An adjacency matrix representing the
            edges in the generated graph.
        Communities (list): The node assignment to communities.
    """
    if community is None:
        # assign a community to each node
        community = np.repeat(list(range(num_classes)),np.ceil(num_nodes/num_classes))

        #np.repeat(list to iterate over, how many times to repeat an item)

        #make sure community has size n
        community = community[0:num_nodes]
        # just in case repeat repeated too many

    communities = community.copy()

    # make it a collumn vector
    community = np.expand_dims(community,1)

    # generate a boolean matrix indicating whether
    # two nodes share a community
    # this is a smart way to generate a section graph
    intra = community == community.T
    inter = community != community.T# we can also use np.logical not

    random = np.random.random((num_nodes,num_nodes))
    tri = np.tri(num_nodes,k=-1).astype(bool)

    intergraph = (random < p_intra) * intra * tri
    # this creates a matrix that only has trues where
    # random< p_intra, they are in intra, and along half the matrix
    # (if it were the whole matrix it would be double the edges we want)
    intragraph = (random < p_inter) * inter * tri# same thing here
    graph = np.logical_or(intergraph,intragraph)
    graph = graph*1# this converts it to a int tensor
    graph += graph.T
    return graph,communities

def epsilon_nn_graph(n_nodes, epsilon_same, epsilon_diff, mu):
    """
    Generate a graph with nodes sampled from a unit square and edges based on distance criteria.

    The function performs the following steps:
    1. Randomly sample points within the unit square.
    2. Assign classes (0 or 1) to the sampled points.
    3. Connect nodes based on the distance between them and their classes:
       - Nodes of the same class are connected if the distance is less than or equal to epsilon_same.
       - Nodes of different classes are connected if the distance is less than or equal to epsilon_diff.
    4. Generate a random normal vector and ensure its norm is not zero.
    5. Calculate the train_b matrix based on the distance `mu`, random vectors, and the node classes.

    Parameters:
    - n_nodes (int): Number of nodes to be sampled.
    - epsilon_same (float): Distance threshold for connecting nodes of the same class.
    - epsilon_diff (float): Distance threshold for connecting nodes of different classes.
    - mu (float): Scalar representing the distance of data clouds from the origin.

    Returns:
    - tuple: A tuple containing:
        1. train_b (numpy.ndarray): Matrix representing the transformed data points.
        2. classes (numpy.ndarray): Array containing the classes (0 or 1) of the nodes.
        3. edges (torch.Tensor): Tensor containing the edges of the graph.s`.
    """
    # Step 1: Sample points within unit square
    points = np.random.rand(n_nodes, 2)
    
    # Step 2: Assign classes randomly
    classes = np.random.choice([0, 1], size=n_nodes)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(n_nodes):
        G.add_node(i, pos=points[i], class_=classes[i])
    
    # Step 3: Connect nodes based on the criteria
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            distance = np.linalg.norm(points[i] - points[j])
            if classes[i] == classes[j] and distance <= epsilon_same:
                edges.append((i, j))
            elif classes[i] != classes[j] and distance <= epsilon_diff:
                edges.append((i, j))
    
    random_vec = np.random.normal(0,1/10,(10))
    while np.linalg.norm(random_vec) ==0:
        random_vec = np.random.normal(0,1/10,(10))
    random_vec = random_vec/np.linalg.norm(random_vec)
    
    random_vec *= 1.0
    # obtains the random normal vector u how far our clouds are from the origin

    train_z = np.random.normal(0,.2,(n_nodes,10))
    # obtains the random noise vector i presume
    train_v = classes # puts the groups into a format for the equations

    perms = generate_orthogonal_vecs(2,10)
    #print(communities)
    #print(perms)
    dist = mu
    train_b = np.zeros((n_nodes,10))
    for i in range(n_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@random_vec) + train_z[i]/np.sqrt(10)
    return train_b,classes,torch.Tensor(edges).to(torch.long)

def generate_sbm_with_triads(n, lamb,mu, p_motif,motif_type):
    """
    Generate an SBM graph with triadic closure.
    
    Parameters:
    - n: number of nodes
    - block_sizes: list of block sizes
    - block_matrix: matrix of connection probabilities between blocks
    - p_triad: probability of closing a triad
    
    Returns:
    - G: Generated graph
    """
    block_sizes = [n//2,n//2]
    c_in = 10+lamb*np.sqrt(10) # c_in/c_out as described in the equations
    c_out = 10-lamb*np.sqrt(10)
    p_in = c_in/n # compiles these to pass into the SSBM
    p_out = c_out/n
    block_matrix = [[p_in,p_out],[p_out,p_in]]
    train_adj, train_communities = generate_ssbm(n,2,p_in,p_out)
    G = nx.from_numpy_array(train_adj).to_directed()
    
    if motif_type == "triad":
        # Identify potential triads
        potential_triads = []
        for i, j in G.edges():
            neighbors_i = set(G.neighbors(i))
            neighbors_j = set(G.neighbors(j))
            common_neighbors = neighbors_i.intersection(neighbors_j) - {i, j}
            for k in common_neighbors:
                if not G.has_edge(i, k):
                    potential_triads.append((i, k))
        
        # Close triads with probability p_triad
        for i, k in potential_triads:
            if np.random.rand() < p_motif:
                G.add_edge(i, k)
    elif motif_type == "rectangle":
        # Identify potential rectangles
        potential_rectangles = []
        for i, j in G.edges():
            for k in G.neighbors(j):
                for l in G.neighbors(k):
                    if G.has_edge(l, i) and not G.has_edge(i, k) and not G.has_edge(i, l):
                        potential_rectangles.append((i, k, l))
        
        # Close rectangles with probability p_motif
        for i, k, l in potential_rectangles:
            if np.random.rand() < p_motif:
                G.add_edge(i, k)
                G.add_edge(i, l)
    elif motif_type == "pentagon":
        # Identify potential pentagons
        potential_pentagons = []
        for i, j in G.edges():
            for k in G.neighbors(j):
                for l in G.neighbors(k):
                    for m in G.neighbors(l):
                        if G.has_edge(m, i) and not G.has_edge(i, k) and not G.has_edge(i, l) and not G.has_edge(i, m):
                            potential_pentagons.append((i, k, l, m))
        
        # Close pentagons with probability p_motif
        for i, k, l, m in potential_pentagons:
            if np.random.rand() < p_motif:
                G.add_edge(i, k)
                G.add_edge(i, l)
                G.add_edge(i, m)
    edge_index = torch.tensor(list(G.edges()))

    random_vec = np.random.normal(0,1/10,(10))
    while np.linalg.norm(random_vec) ==0:
        random_vec = np.random.normal(0,1/10,(10))
    random_vec = random_vec/np.linalg.norm(random_vec)
    
    random_vec *= 1.0
    # obtains the random normal vector u how far our clouds are from the origin

    train_z = np.random.normal(0,.2,(n,10))
    # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations

    perms = generate_orthogonal_vecs(2,10)
    #print(communities)
    #print(perms)
    dist = mu
    train_b = np.zeros((n,10))
    for i in range(n):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@random_vec) + train_z[i]/np.sqrt(10)
    return train_b,train_v, edge_index

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
    edge_set = set()

    #ensure that there are no duplicate edges
    np_edge_data = edge_list.numpy().T
    for (a,b) in np_edge_data:
        if a <= b:
            edge_set.add((a,b))
        else:
            edge_set.add((b,a))
    new_edges = []
    for (a,b) in edge_set:
        new_edges.append([a,b])
    np_edge_data = np.array(new_edges)

    edge_df = pd.DataFrame(np_edge_data)
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

    # print(f"Percent of self_edges is {self_edges.sum()/len(self_edges)} Hyper edges is {hyper_edges/len(self_edges)}")
    data.edge_index = torch.tensor(np_edge_data).long()
    np_edge_data = np.hstack([np_edge_data,np_edge_data[::-1]])
    # print(np_edge_data.shape)
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

def run_dataset(name,t,mu,model_type):
    """Generates the dataset and model to be trained

    Args:
        name (string): Name of dataset
        t (string): Type of scrambling to do ("edges","features","both")
        mu (float): Feature separation parameter
        model_type (string): The name of the model to be run

    Returns:
        Tuple: A tuple containing the normal accuracy and the scrambled accuracy
    """

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
    elif name == "Epsilon":
        # x,y,edges = hSBM(100,.015,.012,.008,mu,.05)
        # print(x.shape,y.shape,edges.shape)
        x,y,edges = epsilon_nn_graph(1000,.1,.05,mu)
        x = torch.Tensor(x).float()
        y = torch.from_numpy(y).long().reshape(-1)
        # edges = torch.from_numpy(edges)
        perm = torch.randperm(1000)
        train_mask = torch.zeros(1000)
        train_mask[:200] = 1
        train_mask = train_mask[perm]
        test_mask = 1-train_mask
        val_mask = 1-train_mask

        dataset = [Data(x=x,y=y,edge_index=edges.T,train_mask=train_mask.bool(),val_mask=val_mask.bool(),test_mask=test_mask.bool())]
        dataset = GraphDataset(dataset)
        dataset.num_classes = 2
    elif name == "Heirarchical":
        x,y,edges = hSBM(100,.015,.012,.008,mu,.05)
        # print(x.shape,y.shape,edges.shape)
        # x,y,edges = epsilon_nn_graph(1000,.1,.05,mu)
        x = torch.Tensor(x).float()
        y = torch.from_numpy(y).long().reshape(-1)
        # edges = torch.from_numpy(edges)
        perm = torch.randperm(1000)
        train_mask = torch.zeros(1000)
        train_mask[:200] = 1
        train_mask = train_mask[perm]
        test_mask = 1-train_mask
        val_mask = 1-train_mask

        dataset = [Data(x=x,y=y,edge_index=edges.T,train_mask=train_mask.bool(),val_mask=val_mask.bool(),test_mask=test_mask.bool())]
        dataset = GraphDataset(dataset)
        dataset.num_classes = 2
    elif name == "Triadic":
        x,y,edges = generate_sbm_with_triads(1000,.6,mu,.3,"triad")
        # print(x.shape,y.shape,edges.shape)
        # x,y,edges = epsilon_nn_graph(1000,.1,.05,mu)
        x = torch.Tensor(x).float()
        y = torch.from_numpy(y).long().reshape(-1)
        # edges = torch.from_numpy(edges)
        perm = torch.randperm(1000)
        train_mask = torch.zeros(1000)
        train_mask[:200] = 1
        train_mask = train_mask[perm]
        test_mask = 1-train_mask
        val_mask = 1-train_mask

        dataset = [Data(x=x,y=y,edge_index=edges.T,train_mask=train_mask.bool(),val_mask=val_mask.bool(),test_mask=test_mask.bool())]
        dataset = GraphDataset(dataset)
        dataset.num_classes = 2




    data = dataset[0]
    data.y = data.y.reshape(-1)
    # print(data.y)
    if not is_transformer:
        if name != "Synthetic":
            model = GNN(data.num_features,32,dataset.num_classes,model_type)
        else:
            model = GNN(10,32,2,type=model_type)
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



        model = GNN(data.num_features,32,dataset.num_classes,type=model_type)
        optimizer = torch.optim.Adam(params = model.parameters(), lr = .01,weight_decay=5e-4)
        data = data.cuda()
        model = model.cuda()
        scrambled = assess(False,model,optimizer,data)
        return normal,scrambled
    else:
        model = GraphTransformer(in_size = data.num_features,
                                    num_class=dataset.num_classes,
                                    d_model = 10,
                                    dim_feedforward=2*10,
                                    num_layers=2,
                                    gnn_type='gcn',
                                    use_edge_attr=False,
                                    num_heads = 10,
                                    in_embed=False,
                                    se="gnn",
                                    use_global_pool=False
                                    )
        optimizer = torch.optim.Adam(params = model.parameters(), lr = .01,weight_decay=5e-4)
        data = data.cuda()
        model = model.cuda()

        normal = assess(False,model,optimizer,data,transformer=True)
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



        model = GraphTransformer(in_size = data.num_features,
                                    num_class=dataset.num_classes,
                                    d_model = 10,
                                    dim_feedforward=2*(10),
                                    num_layers=2,
                                    gnn_type='gcn',
                                    use_edge_attr=False,
                                    num_heads = 10,
                                    in_embed=False,
                                    se="gnn",
                                    use_global_pool=False
                                    )
        optimizer = torch.optim.Adam(params = model.parameters(), lr = .01,weight_decay=5e-4)
        data = data.cuda()
        model = model.cuda()
        scrambled = assess(False,model,optimizer,data,transformer = True)
        return normal,scrambled

# Benchmark datasets implemented
#"Cora","Citeseer","Pubmed","DBLP","Computers","Photo","Flickr","GitHub","FacebookPagePage"

# Synthetic datasets implemented
# "Triadic","Heirarchical","Epsilon"
datasets = ["Triadic", "Heirarchical","Epsilon"]
# Specify if you are wanting to use the synthetic datasets
is_synthetic = True 

# If we want to use a transformer
is_transformer = False

runs = 10
typ = ["edges","features", "both"]
models = ["SAGE","GAT","GCN"]

# Loop through and store any variants of models that we want
for model in models:
    # Loop through the types of scrambling
    for t in typ:
        # Run through various values of mu if it is synthetic
        if is_synthetic:
            # Loop through each dataset
            for i,a in enumerate(datasets):
                li = []
                # Loop through mu to get a range
                for mu in tqdm(np.linspace(0,1,20)):
                    normal_avg = 0
                    scrambled_avg = 0
                    for j in range(runs):
                        normal,scrambled = run_dataset(a,t,mu,model)
                        normal_avg += normal/runs
                        scrambled_avg += scrambled/runs
                    li.append([(normal_avg),(scrambled_avg),mu])
                    print(f"Finished with dataset {a} {mu} {normal_avg} {scrambled_avg}")
                np.savetxt(f"{a}_sbm_{model}_{t}.txt",np.array(li))
        else:
            li = []
            # Loop through every dataset
            for i,a in enumerate(datasets):
                #take the average over several runs
                normal_avg = 0
                scrambled_avg = 0
                for j in range(runs):
                    normal,scrambled = run_dataset(a,t,0,model)
                    normal_avg += normal/runs
                    scrambled_avg += scrambled/runs
                li.append([(normal_avg),(scrambled_avg)])
                print(f"Finished with dataset {0} {normal_avg} {scrambled_avg}")
            np.savetxt(f"benchmarks_{model}.txt",np.array(li))