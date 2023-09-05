# Synthetic-Graphs
---
This is the repository where we can put our code for the Synthetic graph stuff. In this project we investigate how various properties of training data affect the performance and convergence of graph neural network architectures.
In particular we study how attributes such as Degree Distribution, Heterophily, and abundance of Graphlets play a role in the performance of these GNNs. We find that GNNs perform well under heterophily given that the degree of heterophily is great enough. Furthermore, GNNs perform better on scale free graphs and graphlets are not helpful for GNN performance.
## Directory
---
- The data folder contains all the data obtained from test runs.<br>
- The docs folder contains the documentation.<br>
- The result folder contains important plots generated by the test runs.<br>
- The script folder contains all the code to analyze data and produce the plots. It also contains the jupyter notebooks implementation of various SBMs.<br>
- The src folder contains modules that generate synthetic data, various Neural Network Architectures to operate on said data, and miscellaneous functions. These are found in the generate_data.py, models.py, and utils.py respectively.<br>
- The tests folder contains models that test each step of the process and are able to plot the results found in data<br>
## Particular things to look at with plots(Headlines)
- Higher order structure in graphs are detrimental to Graph Neural Network performance
- Transformers and SAGE are able to utilize negative edge information(heterophily) more successfully than GCN and GAT are able to.
- Powerlaw edge distributions outperform binomial edge distributions.
- Some models are able to always exceed or at least approximate a Neural Network, the same is not true for spectral methods
- Weak recovery phase transitions take on a particular shape for each model and we able to see that in each plot
- We can very acutely compare the dissasortative and assortative cases.

