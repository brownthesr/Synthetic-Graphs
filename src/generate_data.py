"""
This is to store all the data generation functions
"""
from itertools import permutations,combinations_with_replacement
import numpy as np
from scipy.stats import poisson
from scipy.stats import bernoulli

def gen_stoney_sbm(num_nodes,power_law,num_classes,p_intra,p_inter):
    """Generates a DC_SBM the way stoney proposed

    Args:
        num_nodes (int): The number of nodes.
        power_law (double): The type of power law distribution,
            usually between 2 and 3.
        num_classes (int): The number of classes.
        p_intra (float): The probability of having connections
            within a class.
        p_inter (float): The probability of having connections 
            between classes.

    Returns:
        list: An adjacency matrix of the new graph.
        list: The community assignments.
    """
    # assign labels
    community = np.repeat(list(range(num_nodes)),np.ceil(num_nodes/num_classes))
    communities = community[:num_nodes]
    phi = np.ones(num_nodes)-np.random.power(power_law,num_nodes)

    phis = np.outer(phi,phi)
    intra = np.expand_dims(communities,1) == np.expand_dims(communities,1).T
    inter = ~ intra
    intra = intra * p_intra
    inter = inter * p_inter
    mul = intra + inter
    phis = phis * mul

    random = np.random.random((num_nodes,num_nodes))
    tri = np.tri(num_nodes,k=-1).astype(bool)

    graph = (random < phis)* tri
    graph = graph*1
    graph += graph.T

    return graph,communities

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

def generate_perms_cube(num_classes,num_feat):
    """Generates points on the corners of a hypercube.

    It uses the number of ways that [0,1] can be arranged in num_feat
    different ways to generate points on a hypercube. It just takes
    the first num_classes of points to use for calculation.

    Args:
        num_classes (int): The number of classes or number
            of points that we want to return.
        num_feat (int): The number of dimensions that we are
            allowed to use.

    Returns:
        vecs (list): A list of vectors that corrospond to
            points on a hypercube
    """
    assert num_classes <= 2**num_feat
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

def generate_csbm(avg_degree,degree_separation,feature_separation,num_features,num_nodes,num_classes):
    """Creates a Contextual Stochastic Block model

    It takes the avg degree, degree separation, and feature separation
    to generate a SBM with features corrosponding to classes.

    Args:
        avg_degree (int): The average degree of all the nodes.
        degree_separation (float): The separation constrain for
            edges in classes and edges between classes.
        features_separation (float): A measure of how far apart
            the means are.
        num_features (int): The number of features to generate.
        num_nodes (int): The number nodes to generate.
        num_classes (int): The number of classes to generate

    Returns:
        Adjacency matrices for a training and testing set.
        Features for training and testing sets.
        Labels for training and testing sets.
    """
    c_in = avg_degree+degree_separation*np.sqrt(avg_degree) # c_in/c_out as described in the equations
    c_out = avg_degree-degree_separation*np.sqrt(avg_degree)
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes

    rand_vec = np.random.normal(0,1/num_features,(num_features))
    # obtains the random normal vector u how far our clouds are from the origin

    train_adj, train_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out)
    # obtains the graph structure
    train_z = np.random.normal(0,.2,(num_nodes,num_features))
    # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations

    perms = generate_orthogonal_vecs(num_classes,num_features)
    #print(communities)
    #print(perms)
    dist = np.sqrt(feature_separation/num_nodes)
    train_b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@rand_vec) + train_z[i]/np.sqrt(num_features)

    # recompute all this but for a test set
    test_adj, test_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out)
    # change graph structure
    test_z = np.random.normal(0,.2,(num_nodes,num_features))
    # change the noise vector, but don't change the community centers
    test_b = np.zeros((num_nodes,num_features))
    test_v = test_communities
    for i in range(num_nodes):
        test_b[i] = dist*(np.diag(perms[test_v[i]])@rand_vec) + test_z[i]/np.sqrt(num_features)

    return train_adj,train_b,train_communities, test_adj,test_b,test_communities

def generate_csbm_modified(avg_degree,degree_separation,
        origin_distance,num_features,num_nodes,num_classes):
    """Creates a Modified Contextual Stochastic Block Model

    This is very similar to the original implementation of the CSBM
    but it changes the metric to measure distance between means to be
    distance from the origin

    Args:
        avg_degree (int): The average degree of all the nodes.
        degree_separation (float): The separation constrain for
            edges in classes and edges between classes.
        origin_distance (float): How far from the origin the means
            are.
        num_features (int): The number of features to generate.
        num_nodes (int): The number nodes to generate.
        num_classes (int): The number of classes to generate

    Returns:
        Adjacency matrices for a training and testing set.
        Features for training and testing sets.
        Labels for training and testing sets.
    """
    c_in = avg_degree+degree_separation*np.sqrt(avg_degree) # c_in/c_out as described in the equations
    c_out = avg_degree-degree_separation*np.sqrt(avg_degree)
    p_in = c_in/num_nodes # compiles these to pass into the SSBM
    p_out = c_out/num_nodes

    random_vec = np.random.normal(0,1/num_features,(num_features))
    # obtains the random normal vector u how far our clouds are from the origin

    train_adj, train_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out)
     # obtains the graph structure
    train_z = np.random.normal(0,.2,(num_nodes,num_features))
    # obtains the random noise vector i presume
    train_v = train_communities # puts the groups into a format for the equations

    perms = generate_orthogonal_vecs(num_classes,num_features)
    #print(communities)
    #print(perms)
    dist = origin_distance
    train_b = np.zeros((num_nodes,num_features))
    for i in range(num_nodes):
        train_b[i] = dist*(np.diag(perms[train_v[i]])@random_vec) + train_z[i]/np.sqrt(num_features)

    # recompute all this but for a test set
    test_adj, test_communities = generate_ssbm(num_nodes,num_classes,p_in,p_out)
    # change graph structure
    test_z = np.random.normal(0,.2,(num_nodes,num_features))
    # change the noise vector, but don't change the community centers
    test_b = np.zeros((num_nodes,num_features))
    test_v = test_communities
    for i in range(num_nodes):
        test_b[i] = dist*(np.diag(perms[test_v[i]])@random_vec) + test_z[i]/np.sqrt(num_features)

    return train_adj,train_b,train_communities, test_adj,test_b,test_communities

def dc_sbm_adj_edges(num_nodes,num_groups,communities,degree_distribution,group_edge_avg):
    """Generates the edges in a DC_SBM

    Manually iterates through all of the communities and generates
    edges according to the degree distribution and group edge avg.

    Args:
        num_nodes (int): The number of nodes in the graph.
        num_groups (int): The number of clusters.
        communities (list): The community assignments.
        degree_distribution (list): List of the expected degree
            of any given node.
        group_edge_avg (list): The expected number of edges
            between groups.

    Returns:
        adj (list): An adjacency matrix of the DC_SBM.
    """
    N = num_nodes
    w = group_edge_avg
    theta = np.array(degree_distribution)
    adj = np.zeros((N,N))
    total_edges = 0
    expected_degrees = degree_distribution
    actual_degrees = np.zeros_like(expected_degrees)
    for i in range(num_groups):
        for j in range(i+1):
            # in this we calculuate the expected number of edges between any two groups
            if i != j:
                num_edges = poisson.rvs(w[i,j])
            else:
                num_edges = poisson.rvs(w[i,j])
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
                if ([start_actuals[k],end_actuals[k]] in pairs
                        or start_actuals[k] == end_actuals[k]
                        or [end_actuals[k],start_actuals[k]] in pairs):
                    new_pair = [start_actuals[k],end_actuals[k]]
                    while (new_pair in pairs
                         or new_pair[0] == new_pair[1]
                         or new_pair.reverse() in pairs):
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

def dc_ssbm(num_nodes,num_groups,separation):
    """Generates a Degree Corrected Stochastic Block Model with Features

    We first assign communities to each node, then we assign them
    degrees(generated randomly) from a powerlaw distribution. Following
    this we obtain w(a parameter used to calculate edges in the SBM)
    using the degrees. Then we obtain the adjacency matrix and the
    features using other methods.

    This function copmutes the degree distribution and group
    assignments prior to letting dc_sbm_adj_edges() calculate
    the actual edges.

    Args:
        num_nodes (int): The number of nodes for the graph.
        num_groups (int): The number of groups for the graph.
        separation (float between [0,1]): The interpolation constant
            between completely random and completely planted
            graphs. A value of 0 corrosponds to a completely
            random graph, a value of 1 corrosponds to a
            completely planted graph.

    Returns:
        adj (list): An adjacency matrix for our new graph.
        communities (list): The community assignments for
            the graph.
    """
    # assign communities to all of our nodes
    community = np.repeat(list(range(num_groups)),np.ceil(num_nodes/num_groups))
    communities = community[:num_nodes]

    # get expected degrees for each node according to a powerlaw distribution
    degree_probs = np.arange((num_nodes)-1) + 1# here we make the assumption that the degree of
                                            # a node won't surpass the number of nodes
                                            # within it's respective group
    degree_probs = 1/(degree_probs*degree_probs)
    degree_probs = degree_probs/degree_probs.sum()
    degrees = np.random.choice(np.arange(num_nodes-1) + 1, size = (num_nodes), p = degree_probs)

    # we sort degrees according to their groups then obtain total degrees for each group
    theta = degrees.copy()
    theta = theta.reshape(num_groups,num_nodes//num_groups)
    group_deg = theta.sum(axis=1)
    theta = np.sort(theta)
    theta = theta.flatten()

    num_edges = sum(group_deg)/2

    group_deg = np.array(group_deg)# we want it to be half, empiraclly this
    # makes our model work how it is supposed to

    # obtain w as a mix between planted and random graphs
    density_enhancer = 1.0# Here we base the density off of the total degrees of each group just to
    # ensure that edges are assigned prooportionally. By increasing this density we ensure that
    # all the nodes receive an edge
    w_planted = np.diag(group_deg)*density_enhancer
    w_random = np.outer(group_deg,group_deg)/(2*num_edges)*density_enhancer
    # not sure if this is normalized as we want it
    w = separation*w_planted + (1-separation)*w_random

    # obtain our adjacency matrix along with the corrosponding features
    adj = dc_sbm_adj_edges(num_nodes,num_groups,communities,theta, w)
    #features = get_features(num_nodes,num_features,num_groups,mu,communities)

    #README the feature generation does not take class size into account,
    # or community ordering. If you wanted
    # to specify this you could.
    return adj,communities

def generate_dc_ssbm(num_nodes,num_groups,num_features,separation,origin_distance):
    """Generates the Training and Testing DC_SBMs

    This function utilizes the DC_SBM to actually generate
    adjacency matrices. It also computes the feature vectors
    for any given node in the graph, but keeps the means the
    same.

    Args:
        num_nodes (int): The number of nodes in the graph.
        num_groups (int): The number of groups to generate.
        num_features (int): The number of features to generate.
        separation (float): The interpolation constant between
            planted and random graphs.
        origin_distance (float): The distance of the means from
            the origin.

    Returns:
        Adjacency matrices for a training and testing set.
        Features for training and testing sets.
        Labels for training and testing sets.
    """
    rand_vec = np.random.normal(0,1/num_features,(num_features))
    # obtains the random normal vector rand_vec, how far our clouds are from the origin
    perms = generate_orthogonal_vecs(num_groups,num_features)
    # how we want to permute the random vector rand_vec

    def get_feature_vecs(perm_idx,noise):
        """Generates features vectors

        Generates feature vectors along a given mean
        and distance from the origin. But can vary how
        the noise is applied to these vectors and what
        which means they are applied too(because of class)

        Args:
            perm_idx (list): The permutation index or the class
                assignments. Basically tells us which mean to use.
            noise (list): The noise we will apply to each of the
                means.

        Returns:
            b (list): A list of feature vectors for all the nodes
                in the graph.
        """
        features = np.zeros((num_nodes,num_features))
        for i in range(num_nodes):
            features[i] = origin_distance*(np.diag(perms[perm_idx[i]])@rand_vec)+ noise[i]/np.sqrt(num_features)
        return features


    train_adj, train_communities = dc_ssbm(num_nodes,num_groups,separation)
    train_noise = np.random.normal(0,.2,(num_nodes,num_features))
    train_features = get_feature_vecs(train_communities,train_noise)

    test_adj, test_communities = dc_ssbm(num_nodes,num_groups,separation)
    test_noise = np.random.normal(0,.2,(num_nodes,num_features))
    test_features = get_feature_vecs(test_communities,test_noise)

    return train_adj,train_features,train_communities, test_adj,test_features,test_communities

def xor_data(num_nodes, feat_dim,log_scaling):
    """Generates XOR data

    Generates binary data in clouds that are orthogonal
    to each other. Additionally each class has two clouds
    one that is positive and one that is negative.

    Args:
        num_nodes (int): The number of nodes
        feat_dim (int): The number of features..
        log_scaling (float): How far we want our clouds
            from the origin. Measured in log10.

    Returns:
        training and test feature vectors
        community assignments
    """
    bern = bernoulli(.5)
    communities = bern.rvs(num_nodes)
    orientation = bern.rvs(num_nodes)
    vecs = generate_orthogonal_vecs(2,feat_dim)
    u = vecs[0]*(10**log_scaling)
    v = vecs[1]*(10**log_scaling)
    std = 1/feat_dim
    X = np.random.normal(0,std,(num_nodes, feat_dim))
    for i in range(num_nodes):
        X[i] =  X[i] + (2*orientation[i] - 1)*((1 - communities[i])* u + communities[i] * v)

    test_x = np.random.normal(0,std,(num_nodes, feat_dim))
    for i in range(num_nodes):
        test_x[i] =  test_x[i] + (2*orientation[i] - 1)* ((
            1 - communities[i])* u + communities[i] * v)
    return X,test_x,communities

def xor_sbm(num_nodes, feat_dim, intra, inter, log_scaling):
    """Generates an XOR_SBM as outlined in "Effects of Graph Convolutions in Deep Networks".

    Args:
        num_nodes (int): The number of nodes.
        feat_dim (int): The number of features.
        intra (float): The probability of having edges
            within a class.
        inter (float): The probability of having edges
            between different classes.
        log_scaling (float): How far we want our feature
            clouds from the origin.

    Returns:
        training and testing features.
        training and testing adjacency matrices.
        community assignments.
    """
    features, test_features, communities = xor_data(num_nodes,feat_dim,log_scaling)
    adj, _ = generate_ssbm(num_nodes,2,intra,inter,communities)
    test_adj, _ = generate_ssbm(num_nodes,2,intra,inter,communities)
    return features, adj, test_features, test_adj, communities
