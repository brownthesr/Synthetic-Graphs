# Paper we have found in the Lit Search:

	Page rank - Proposes a new clustering method: PageRank. It uses cSBM's, but they are not main focus of paper
	cSBM - Proposes a graph generation tactic. They generate a graph with two two distinct clusters of the same size. They use specific hyper parameters to tune structure and feature similarity between clusters. Features are generated independently according to Gaussian random. The whole paper is focused on this new model.
  
 	MixHop - Proposes a new GNN model to learn mixed neighborhoods better. They use "Visibility of minorities in social networks" to generate graphs, but mainly focus on the new model.
	Beyond Homophily - Propose a new model to help mitigate and learn heterophily better. They use "Visibility of minorities in social networks" to generate graphs, but also mainly focus on the new model.
	Visibility of minorities in social networks - They propose a new method of generating graphs. In this method each node is randomly assigned a group according to the prior (p- prior, h_i,j - homophily between two nodes i and j(according to group assignment or features), k - degree of node).  P(i,j) = (h_i,j*k_i)/(sum_(all nodes)(h_l,j*k_l)) is the probability of edge between i and j. In general h is defined on a group basis. They mainly focus on generating the graph structure and not as much on the features.

	SuperGAT - Proposes a variant of the GAT to learn transductive graphs better. They use Networkx to develop a SSBM(Symmetric Stochastic Block Model) to run the model on. They use overlapping Gaussian Distributions to generate features based on class. They are mainly focused on the model

	Covariate Regularized Community Detection in Sparse Graphs - They use a version of SBM to generate random feature vectors with covariates. They specifically focus on sparse graphs.

	New Benchmarks for Learning on Non-Homophilous Graphs- Proposes that GNN's should be trained on specific heterophilous graphs. They only mention synthetic graphs

	Robustness of Community Detection to Random Geometric Perturbations - They show that spectral methods are robust to some types of noise. They use an SBM

# Things we should search for: ?
 