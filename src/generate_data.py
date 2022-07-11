"""
This is to store all the data generation functions
"""
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations,combinations,combinations_with_replacement
from scipy.stats import poisson
from scipy.stats import bernoulli

def gen_stone_sbm(n,a,c,p_intra,p_inter):
    # assign labels
    community = np.repeat(list(range(n)),np.ceil(n/c))
    communities = community[:n]
    phi = np.ones(n)-np.random.power(a,n)
    adj = np.zeros((n,n))

    phis = np.outer(phi,phi)
    intra = np.expand_dims(communities,1) == np.expand_dims(communities,1).T
    inter = ~ intra
    intra = intra * p_intra
    inter = inter * p_inter
    mul = intra + inter
    phis = phis * mul

    random = np.random.random((n,n))
    tri = np.tri(n,k=-1).astype(bool)

    graph = (random < phis)* tri
    graph = graph*1
    graph += graph.T

    return graph,communities

def generate_ssbm(num_nodes,num_classes,p_intra,p_inter,community = None):# generates a symmetric SBM
    """This is similar to the above SBM but in this case it is symmetric"""
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

def generate_perms_cube(num_classes,num_feat):
    """
    Generates where our point clouds will exist relative to the origin, uses a cube
    """
    assert num_classes <= 2**num_feat
    dims = np.ceil(np.log2(num_classes))+1
    vecs = []
    #vec1 = np.ones(num_feat)
    #vecs.append(vec1.copy())
    #vec1[-1] = -1
    #vec1[-2] = -1
    #perms = list(set(permutations(vec1)))


    combs = np.array(list(combinations_with_replacement([0,1],num_feat)))
    total = []
    for i,combination in enumerate(combs):
        total.append(set(permutations(combination)))
    for i in total:
        for j in i:
            vecs.append(j)
    
    #print(perms)
    #for i in perms:
    #    vecs.append(i)
    #print(vecs)
    return np.array(vecs[:num_classes])

def generate_perms_orthogonal(num_classes,num_feat):
    """ 
    same thing, but initializes random orthogonal vectors instead
    """
    vecs = []
    for i in range(num_classes):
        orthogonal = False
        while(not orthogonal):
            rand_vec = np.random.uniform(-10,10,num_feat)
            rand_vec = rand_vec/np.linalg.norm(rand_vec)
            orthogonal = True
            for j in range(len(vecs)):
                angle = np.arccos(rand_vec@vecs[j])
                if(angle < np.pi/2 - .1 or angle > np.pi/2 + .1):
                    orthogonal = False
        vecs.append(rand_vec)
        #print(len(vecs))
    return np.array(vecs)

def generate_cSBM(d,lamb,mu,num_features,num_nodes,num_classes):
    c_in = d+lamb*np.sqrt(d) # c_in/c_out as described in the equations
    c_out = d-lamb*np.sqrt(d) 
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes

    u = np.random.normal(0,1/num_features,(num_features)) # obtains the random normal vector u how far our clouds are from the origin

    train_adj, train_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out) # obtains the graph structure
    train_Z = np.random.normal(0,.2,(num_nodes,num_features)) # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations
    
    perms = generate_perms_orthogonal(num_classes,num_features)
    #print(communities)
    #print(perms)
    dist = np.sqrt(mu/num_nodes)
    train_b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@u) + train_Z[i]/np.sqrt(num_features)
        
    # recompute all this but for a test set
    test_adj, test_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out)# change graph structure
    test_Z = np.random.normal(0,.2,(num_nodes,num_features)) # change the noise vector, but don't change the community centers
    test_b = np.zeros((num_nodes,num_features))
    test_v = test_communities
    for i in range(num_nodes):
        test_b[i] = dist*(np.diag(perms[test_v[i]])@u) + test_Z[i]/np.sqrt(num_features)
    
    return train_adj,train_b,train_communities, test_adj,test_b,test_communities

def generate_cSBM_modified(d,lamb,o_distance,num_features,num_nodes,num_classes):
    c_in = d+lamb*np.sqrt(d) # c_in/c_out as described in the equations
    c_out = d-lamb*np.sqrt(d) 
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes

    u = np.random.normal(0,1/num_features,(num_features)) # obtains the random normal vector u how far our clouds are from the origin

    train_adj, train_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out) # obtains the graph structure
    train_Z = np.random.normal(0,.2,(num_nodes,num_features)) # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations
    
    perms = generate_perms_orthogonal(num_classes,num_features)
    #print(communities)
    #print(perms)
    dist = o_distance
    train_b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@u) + train_Z[i]/np.sqrt(num_features)
        
    # recompute all this but for a test set
    test_adj, test_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out)# change graph structure
    test_Z = np.random.normal(0,.2,(num_nodes,num_features)) # change the noise vector, but don't change the community centers
    test_b = np.zeros((num_nodes,num_features))
    test_v = test_communities
    for i in range(num_nodes):
        test_b[i] = dist*(np.diag(perms[test_v[i]])@u) + test_Z[i]/np.sqrt(num_features)
    
    return train_adj,train_b,train_communities, test_adj,test_b,test_communities

def DC_SBM_adj_edges(num_nodes,num_groups,communities,degree_distribution,group_edge_avg):
    N = num_nodes
    w = group_edge_avg
    theta = np.array(degree_distribution)
    adj = np.zeros((N,N))
    total_edges = 0
    expected_degrees = degree_distribution
    actual_degrees = np.zeros_like(expected_degrees)
    edges_added = 0
    for i in range(num_groups):
        for j in range(i+1):
            # in this we calculuate the expected number of edges between any two groups
            if i != j:
                num_edges = poisson.rvs(group_edge_avg[i,j])
            else:
                num_edges = poisson.rvs(group_edge_avg[i,j])
            total_edges += num_edges

            group1_indices = np.where(communities == i)[0]
            group2_indices = np.where(communities == j)[0]

            start_probs = theta[group1_indices]/theta[group1_indices].sum()
            end_probs = theta[group2_indices]/theta[group2_indices].sum()

            start_actuals = np.random.choice(group1_indices,size=(num_edges),p=start_probs)
            end_actuals = np.random.choice(group2_indices,size=num_edges,p=end_probs)

            # this ensures that no edges repeat themselves
            pairs = []
            for k in range(len(start_actuals)):
                if [start_actuals[k],end_actuals[k]] in pairs or start_actuals[k] == end_actuals[k] or [end_actuals[k],start_actuals[k]] in pairs:
                    #print([start_actuals[k],end_actuals[k]] in pairs,start_actuals[k] == end_actuals[k],[end_actuals[k],start_actuals[k]] in pairs)
                    new_pair = [start_actuals[k],end_actuals[k]]
                    while new_pair in pairs or new_pair[0] == new_pair[1] or new_pair.reverse() in pairs:
                        new_pair = [np.random.choice(group1_indices,size=(1),p=start_probs),
                        np.random.choice(group2_indices,size=1,p=end_probs)]
                    pairs.append(new_pair)
                else:
                    pairs.append([start_actuals[k],end_actuals[k]])

            pairs = np.array(pairs).astype(int)
            if num_edges > 0:
                adj[pairs[:,0],pairs[:,1]] += 1
                adj[pairs[:,1],pairs[:,0]] += 1
                actual_degrees[pairs[:,0]] += 1
                actual_degrees[pairs[:,1]] += 1
    return adj

def DC_SSBM(num_nodes,num_groups,num_features,lamb,mu):
    """
    Generates a Degree Corrected Stochastic Block Model with Features
    
    We first assign communities to each node, then we assign them degrees(generated randomly) from
    a powerlaw distribution. Following this we obtain w(a parameter used to calculate edges in the SBM)
    using the degrees. Then we obtain the adjacency matrix and the features usign other methods
    """
    # assign communities to all of our nodes
    community = np.repeat(list(range(num_groups)),np.ceil(num_nodes/num_groups))
    communities = community[:num_nodes]

    # get expected degrees for each node according to a powerlaw distribution
    p = np.arange((num_nodes)-1) + 1# here we make the assumption that the degree of a node won't surpass the number of nodes
                                            # within it's respective group
    p = 1/(p*p)
    p = p/p.sum() 
    degrees = np.random.choice(np.arange(num_nodes-1) + 1, size = (num_nodes), p = p)

    # we sort degrees according to their groups then obtain total degrees for each group
    theta = degrees.copy()
    theta = theta.reshape(num_groups,num_nodes//num_groups)
    group_deg = theta.sum(axis=1)
    theta = np.sort(theta)
    theta = theta.flatten()
    
    num_edges = sum(group_deg)/2

    group_deg = np.array(group_deg)# we want it to be half, empiraclly this makes our model work how it is supposed to

    # obtain w as a mix between planted and random graphs
    density_enhancer = 1.0# Here we base the density off of the total degrees of each group just to
    # ensure that edges are assigned prooportionally. By increasing this density we ensure that 
    # all the nodes receive an edge
    w_planted = np.diag(group_deg)*density_enhancer
    w_random = np.outer(group_deg,group_deg)/(2*num_edges)*density_enhancer# not sure if this is normalized as we want it
    w = lamb*w_planted + (1-lamb)*w_random

    # obtain our adjacency matrix along with the corrosponding features
    adj = DC_SBM_adj_edges(num_nodes,num_groups,communities,theta, w)
    #features = get_features(num_nodes,num_features,num_groups,mu,communities)

    #README the feature generation does not take class size into account, or community ordering. If you wanted
    # to specify this you could.
    return adj,communities

def generate_DC_SBM(num_nodes,num_groups,num_features,lamb,mu):

    u = np.random.normal(0,1/num_features,(num_features)) # obtains the random normal vector u how far our clouds are from the origin

    train_adj, train_communities = DC_SSBM(num_nodes,num_groups,num_features,lamb,mu) # obtains the graph structure
    train_Z = np.random.normal(0,.2,(num_nodes,num_features)) # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations
    
    perms = generate_perms_orthogonal(num_groups,num_features)
    #print(communities)
    #print(perms)
    dist = mu
    train_b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@u) + train_Z[i]/np.sqrt(num_features)
        
    # recompute all this but for a test set
    test_adj, test_communities = DC_SSBM(num_nodes,num_groups,num_features,lamb,mu)# change graph structure
    test_Z = np.random.normal(0,.2,(num_nodes,num_features)) # change the noise vector, but don't change the community centers
    test_b = np.zeros((num_nodes,num_features))
    test_v = test_communities
    for i in range(num_nodes):
        test_b[i] = dist*(np.diag(perms[test_v[i]])@u) + test_Z[i]/np.sqrt(num_features)
    
    return train_adj,train_b,train_communities, test_adj,test_b,test_communities

def XOR_data(num_nodes, feat_dim,p):
    bern = bernoulli(.5)
    communities = bern.rvs(num_nodes)
    orientation = bern.rvs(num_nodes)
    vecs = generate_perms_orthogonal(2,feat_dim)
    u = vecs[0]*(10**p)
    v = vecs[1]*(10**p)
    std = 1/feat_dim
    X = np.random.normal(0,std,(num_nodes, feat_dim))
    for i in range(num_nodes):
        X[i] =  X[i] + (2*orientation[i] - 1)*((1 - communities[i])* u + communities[i] * v)

    test_X = np.random.normal(0,std,(num_nodes, feat_dim))
    for i in range(num_nodes):
        test_X[i] =  test_X[i] + (2*orientation[i] - 1)*((1 - communities[i])* u + communities[i] * v)
    return X,test_X,communities

def XOR_SBM(num_nodes, feat_dim, intra, inter,p):
    features, test_features, communities = XOR_data(num_nodes,feat_dim,p)
    adj, _ = generate_ssbm(num_nodes,2,intra,inter,communities)
    test_adj, _ = generate_ssbm(num_nodes,2,intra,inter,communities)
    return features, adj, test_features, test_adj, communities