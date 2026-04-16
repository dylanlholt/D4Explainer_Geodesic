# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Context
See CONTEXT.md for project purpose, mathematical background, paper outline, pseudocode, and implementation plan.

## Overview

D4Explainer is a PyTorch implementation of "D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion" (NeurIPS 2023). It trains a diffusion-based model to generate explanations for GNN predictions by learning to denoise perturbed graph adjacency matrices.

## Environment
conda activate d4explainer-geo

## Commands

**Train base GNNs** (must be done before training D4Explainer):
```bash
cd gnns
python ba3motif_gnn.py
python mutag_gnn.py
python nci1_gnn.py
python synthetic_gnn.py --data_name Tree_Cycle
python synthetic_gnn.py --data_name BA_shapes
python tree_grids_gnn.py
python web_gnn.py
python bbbp_gnn.py
```

**Train and evaluate D4Explainer:**
```bash
python main.py --dataset mutag
python main.py --dataset ba3
python main.py --dataset Tree_Cycle
# Valid datasets: BA_shapes, Tree_Cycle, Tree_Grids, cornell, mutag, ba3, bbbp, NCI1
```

**Evaluate additional properties:**
```bash
python -m evaluation.ood_evaluation --dataset mutag
python -m evaluation.robustness --dataset mutag
```

## Architecture

**Two-stage pipeline:**
1. **Base GNN training** (`gnns/`): Dataset-specific GNN classifiers trained and saved to `param/gnns/{dataset}_{gnn_type}.pt`. Each implements `get_pred`, `get_node_pred_subgraph`, and `get_pred_explain` methods.
2. **D4Explainer training** (`explainers/`): A diffusion model trained on top of a frozen GNN to generate explanatory subgraphs.

**Core components:**

- `main.py`: Entry point. Parses args, loads datasets, instantiates `DiffExplainer`, runs training then evaluation.
- `constants.py`: `feature_dict` (input feature dims per dataset) and `task_type` (`nc`=node classification, `gc`=graph classification).
- `explainers/diff_explainer.py` → `DiffExplainer`: Orchestrates training. Loss = BCE denoising loss (`loss_func_bce`) + counterfactual loss (`loss_cf_exp`) weighted by `alpha_cf`. Best model saved to `results/{dataset}/best_model.pth`.
- `explainers/diffusion/pgnn.py` → `Powerful`: The denoising network. Takes a noisy adjacency matrix `A [bsz, N, N]`, node features, mask, and noise level σ; outputs edge scores `[bsz, N, N, 1]`. Built from `PowerfulLayer` (matrix-product message passing) and `FeatureExtractor` (invariant pooling).
- `explainers/diffusion/graph_utils.py`: Tensor/graph conversion utilities. `discretenoise_single` adds discrete Bernoulli noise. `gen_list_of_data_single` generates multi-σ training batches. `graph2tensor`/`tensor2graph` convert between PyG `Data` objects and dense adjacency tensors.
- `explainers/base.py` → `Explainer`: Base class. Loads the frozen GNN from disk. Provides `pack_explanatory_subgraph`, `evaluate_acc`, and `visualize`.
- `datasets/`: Dataset loaders returning PyG `Data` objects for each supported dataset.
- `utils/dataset.py` → `get_datasets`: Central dispatch returning `(train, val, test)` splits. Datasets loaded from `data/` directory (must be downloaded separately).
- `evaluation/`: OOD evaluation using MMD on graph statistics (`ood_evaluation.py`) and perturbation robustness testing (`robustness.py`). Run as modules from repo root.

**Data flow during training:**
1. Graph batch → `graph2tensor` → dense `[bsz, N, N]` adjacency + `[bsz, N, C]` features
2. `gen_list_of_data_single` replicates batch for each σ in `sigma_list`, adds discrete noise
3. `Powerful` model predicts denoised adjacency scores per noise level
4. `tensor2graph` converts predicted scores back to PyG graph for GNN evaluation
5. Total loss = denoising BCE + α × counterfactual loss

**Key hyperparameters** (set via CLI args in `main.py`):
- `--prob_low/prob_high`: Range for uniform σ sampling (default 0.0–0.4)
- `--sigma_length`: Number of σ samples per batch (default 10)
- `--alpha_cf`: Weight of counterfactual loss (default 0.5)
- `--sparsity_level`: BCE positive class weight controlling explanation sparsity (default 2.5)
- `--num_layers`, `--n_hidden`: `Powerful` model depth and width

## Dependencies

Requires `torch==1.10.1`, `torch-geometric==2.0.4`, `numpy==1.24.2`, `pandas==1.5.3`, `networkx==3.0`. See `requirements.txt` for the full list. Datasets must be downloaded from Google Drive to `data/`.
