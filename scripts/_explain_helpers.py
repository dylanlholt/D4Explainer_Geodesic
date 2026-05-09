"""Shared utilities for the visualization scripts.

Loads a trained DiffExplainer checkpoint by reading config.json from the run
directory, reconstructing an args namespace that matches what the model was
trained with. This avoids re-specifying every model-architecture flag.
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from constants import feature_dict, task_type
from explainers import DiffExplainer
from gnns import *  # noqa — register GNN classes so torch.load can unpickle GNN checkpoints
from utils.dataset import get_datasets


# Argparse flags that influence model construction or eval-time behavior.
# Read from config.json; anything missing is taken from CLI defaults.
ARG_KEYS = [
    "dataset", "gnn_type", "task", "feature_in",
    "num_layers", "n_hidden", "layers_per_conv", "normalization",
    "cat_output", "residual", "noise_mlp", "simplified",
    "dropout", "threshold", "alpha_cf",
    "prob_low", "prob_high", "sigma_length",
    "train_batchsize", "test_batchsize",
    "natural_gradient", "nat_grad_eps", "nat_grad_cap", "nat_grad_cf_only",
    "memory_efficient", "use_amp", "gradient_checkpointing",
    "size_bucketed", "max_graph_size",
    "run_name", "root",
]


def load_args_from_run(run_dir, device, noise_list=None):
    """Build an args namespace from results/{dataset}/{run_name}/config.json."""
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.json at {run_dir} — checkpoint likely never ran.")
    with open(cfg_path) as f:
        cfg = json.load(f)

    args = SimpleNamespace()
    for k in ARG_KEYS:
        setattr(args, k, cfg.get(k))
    args.device = device
    args.noise_list = noise_list
    if args.feature_in is None:
        args.feature_in = feature_dict[args.dataset]
    if args.task is None:
        args.task = task_type[args.dataset]
    return args


def assert_checkpoint_exists(run_dir):
    pth = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(pth):
        raise FileNotFoundError(
            f"best_model.pth not found at {run_dir}. Wait for the training run to "
            f"hit at least one is_best epoch, then re-run."
        )


def load_test_dataset(dataset_name):
    _, _, test_dataset = get_datasets(name=dataset_name)
    return test_dataset


def build_explainer(device, dataset_name, gnn_type="gcn"):
    gnn_path = f"param/gnns/{dataset_name}_{gnn_type}.pt"
    return DiffExplainer(device, gnn_path)
