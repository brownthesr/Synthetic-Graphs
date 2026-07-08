# Synthetic-Graphs

Code for studying how the **properties of training data** shape the performance, expressivity, and recoverability of graph neural network (GNN) architectures on community detection tasks.

> 📄 **Publication.** This work has been **accepted to the *SIAM Journal on Applied Mathematics*** (in press, 2026):
> *"Beyond Linear: A Theoretical and Empirical Analysis of Nonlinear GNNs for Community Detection."*
> Drake B. Brown\*, Trevor Garrity\*, Kaden Parker, Jason Oliphant, Stone Carson, Cole Hanson, Dustin Angerhofer, Zachary Evans, Xiao Li, Joseph du Toit, Zachary M. Boyd.
> (\*equal contribution)

## Overview

We generate synthetic graphs with controllable structural properties — degree distribution, heterophily, community structure, and higher-order (graphlet) content — and systematically measure how different GNN architectures respond as those properties vary. Using synthetic data lets us isolate *which* graph properties drive performance, rather than confounding them as real-world datasets do.

The study spans architectures including GCN, GAT, GraphSAGE, and graph transformers, and connects empirical behavior to theory around recoverability thresholds and phase transitions in community detection.

## Key findings

- **Higher-order structure matters.** Graphlet-level structure strongly influences GNN performance, beyond what pairwise edge statistics capture.
- **Architectures exploit heterophily differently.** Transformers and GraphSAGE make better use of heterophily than GCN and GAT.
- **Degree distribution affects recoverability.** Power-law (scale-free) edge distributions yield better performance than binomial ones.
- **Nonlinearity is not free.** Some models always match or approximate a general neural network, while purely spectral methods do not — clarifying the gap between linear and nonlinear GNNs.
- **Phase transitions are model-specific.** The weak-recovery phase transition has a different shape depending on the architecture, in both assortative and disassortative regimes.

## Repository structure

| Path | Contents |
|------|----------|
| `src/` | Core modules: synthetic data generation (`generate_data.py`), model architectures (`models.py`), utilities (`utils.py`) |
| `scripts/` | Experiment drivers and analysis code |
| `plotting_scripts/` | Scripts that reproduce the plots in the paper |
| `results/` | Saved outputs and important plots from experimental runs |
| `compile_synthetic.py` | Aggregates results across runs |
| `plot_structures.py` | Visualizes generated graph structures |
| `environment.yml` | Conda environment specification |
| `setup.py` | Package installation |

## Setup

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate synthetic-graphs

# Install the package (editable)
pip install -e .
```

## Usage

```bash
# Generate synthetic graphs and run architecture comparisons
python compile_synthetic.py

# Reproduce paper figures
python plot_structures.py
# additional figure scripts live in plotting_scripts/
```

See `scripts/` for the individual experiment configurations used in the paper.

## Citation

If you use this code, please cite:

```bibtex
@article{brown2026beyond,
  title   = {Beyond Linear: A Theoretical and Empirical Analysis of Nonlinear GNNs for Community Detection},
  author  = {Brown, Drake B. and Garrity, Trevor and Parker, Kaden and Oliphant, Jason and Carson, Stone and Hanson, Cole and Angerhofer, Dustin and Evans, Zachary and Li, Xiao and du Toit, Joseph and Boyd, Zachary M.},
  journal = {SIAM Journal on Applied Mathematics},
  year    = {2026},
  note    = {In press}
}
```
