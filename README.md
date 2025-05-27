# PF-AGCN: Adaptive Graph Convolutional Network for Protein Function Prediction
Official implementation of **PF-AGCN**, a dual-graph model integrating hierarchical GO term relationships and protein-protein interactions (PPIs) for accurate function prediction.

## Key Features
- **Dual-Graph Learning**: Combines a **function graph** (Gene Ontology hierarchies) and **protein graph** (contact map-based PPIs)
- **Adaptive GCN**: Preserves native biological graph structures while learning refined representations
- **Dual-Branch Module**: Fuses ESM protein language model with dilated causal CNNs for global-local feature synergy
- **State-of-the-Art Performance**: Outperforms baselines on CAFA-style benchmarks across BP/MF/CC ontologies
![Model Architecture](docs/architecture.png)
[Project Page](https://github.com/smyang107/PFAGCN/) 
# dataset
data available at: http://deepgo.bio2vec.net/data/deepgo/data.tar.gz

# running
Download the data file.
Install torch
run main_PFGCN.py

## Installation
```bash
git clone https://github.com/smyang107/PFGCN.git
cd PFGCN

## Create conda environment
conda create -n pf-agcn python=3.8
conda activate pf-agcn

## Install ESM (requires PyTorch 1.11+)
git clone https://github.com/facebookresearch/esm
cd esm && pip install -e . 
