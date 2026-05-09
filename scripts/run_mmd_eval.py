"""Run OOD MMD evaluation for a single (dataset, run_name) checkpoint.

Reuses the explainer's `explain_evaluation` to produce predicted explanation
graphs from the saved best_model.pth, then evaluates degree / cluster /
spectral MMD against the original test graphs via the existing
`eval_graph_list` helper.

Output is JSON; bundle two runs (baseline vs natgrad) in a wrapper for
side-by-side reporting. Usage:

    python -m scripts.run_mmd_eval --dataset Tree_Cycle --run_name baseline
    python -m scripts.run_mmd_eval --dataset Tree_Cycle --run_name natgrad

Defaults to 50 test examples (matching D4Explainer's reported eval). Use
--num_test 100 (etc.) to widen the bound.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# numpy._core shim for cross-version checkpoint loading (mirrors verify_sparsity_direction.py)
import numpy as np
import sys as _sys
if not hasattr(np, "_core"):
    import numpy.core as _npcore
    _sys.modules["numpy._core"] = _npcore
    _sys.modules["numpy._core.multiarray"] = _npcore.multiarray
    _sys.modules["numpy._core.numeric"] = _npcore.numeric

from constants import feature_dict, task_type
from evaluation.in_distribution.ood_stat import eval_graph_list
from explainers import DiffExplainer
from gnns import *  # noqa: F401,F403 — required so torch.load can resolve pickled classifier classes
from utils.dataset import get_datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--run_name", required=True, help="Subdir under results/{dataset}/")
    p.add_argument("--root", default="results")
    p.add_argument("--num_test", type=int, default=50)
    p.add_argument("--gnn_type", default="gcn")
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional JSON output path")
    return p.parse_args()


def main():
    cli = parse_args()

    # Pull the per-checkpoint config so model dims line up.
    cfg_path = os.path.join(cli.root, cli.dataset, cli.run_name, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    args = SimpleNamespace(**cfg)
    # Override paths/flags for this eval invocation.
    args.root = cli.root
    args.run_name = cli.run_name
    args.dataset = cli.dataset
    args.gnn_type = cli.gnn_type
    args.cuda = cli.cuda
    args.device = torch.device(f"cuda:{cli.cuda}" if torch.cuda.is_available() else "cpu")
    args.feature_in = feature_dict[cli.dataset]
    args.task = task_type[cli.dataset]
    args.noise_list = None

    _, _, test_dataset = get_datasets(name=cli.dataset)
    test_loader = DataLoader(
        dataset=test_dataset[: cli.num_test], batch_size=1, shuffle=False, drop_last=False
    )

    gnn_path = f"param/gnns/{cli.dataset}_{cli.gnn_type}.pt"
    explainer = DiffExplainer(args.device, gnn_path)

    test_graphs, pred_graphs = [], []
    for graph in test_loader:
        graph.to(args.device)
        exp_subgraph, _, _, _ = explainer.explain_evaluation(args, graph)
        test_graphs.append(to_networkx(graph, to_undirected=True))
        pred_graphs.append(to_networkx(exp_subgraph, to_undirected=True))

    mmd = eval_graph_list(test_graphs, pred_graphs, methods=["degree", "cluster", "spectral"])
    summary = {
        "dataset": cli.dataset,
        "run_name": cli.run_name,
        "num_test": cli.num_test,
        "deg": float(mmd["degree"]),
        "clus": float(mmd["cluster"]),
        "spec": float(mmd["spectral"]),
        "sum": float(mmd["degree"] + mmd["cluster"] + mmd["spectral"]),
    }

    print(json.dumps(summary, indent=2))
    if cli.out:
        os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
        with open(cli.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
