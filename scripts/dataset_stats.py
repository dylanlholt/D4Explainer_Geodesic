"""Print node/edge distribution stats and predicted Powerful-net activation memory
for each D4Explainer dataset, using paper Table 7 hyperparameters.

Run from repo root:
    python -m scripts.dataset_stats
    python -m scripts.dataset_stats --root data/ --datasets BA_shapes Tree_Cycle
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.dataset import get_datasets

# Paper Table 7: (n_hidden, num_layers, train_batchsize). sigma_length is fixed at 10.
PAPER_HP = {
    "BA_shapes":  (64,  6, 4),
    "Tree_Cycle": (64,  6, 32),
    "Tree_Grids": (128, 8, 32),
    "cornell":    (128, 6, 4),
    "ba3":        (128, 6, 32),
    "bbbp":       (128, 6, 16),
    "NCI1":       (128, 6, 32),
    "mutag":      (64,  6, 2),
}
SIGMA_LENGTH = 10
DTYPE_BYTES = 4  # float32


def split_stats(dataset):
    """Return (n_graphs, node_counts, edge_counts) for a single split."""
    nodes, edges = [], []
    for d in dataset:
        nodes.append(int(d.num_nodes))
        edges.append(int(d.edge_index.size(1)) // 2)  # undirected edge count
    return len(dataset), np.array(nodes), np.array(edges)


def fmt_dist(arr):
    if len(arr) == 0:
        return "—"
    pcts = np.percentile(arr, [50, 95, 99])
    return f"{arr.min()}/{int(pcts[0])}/{int(pcts[1])}/{int(pcts[2])}/{arr.max()}"


def peak_cat_gb(bsz, n_max, n_hidden, num_layers):
    """Powerful's per-layer concat tensor: [bsz * sigma, N, N, hidden * num_layers]."""
    bytes_ = bsz * SIGMA_LENGTH * (n_max ** 2) * (n_hidden * num_layers) * DTYPE_BYTES
    return bytes_ / 1024**3


def report(name, root):
    try:
        train, val, test = get_datasets(name, root=root)
    except Exception as e:
        return f"  [skip] {name}: {type(e).__name__}: {e}"

    n_tr, nodes_tr, edges_tr = split_stats(train)
    n_va, nodes_va, edges_va = split_stats(val)
    n_te, nodes_te, edges_te = split_stats(test)
    nodes_all = np.concatenate([nodes_tr, nodes_va, nodes_te])
    edges_all = np.concatenate([edges_tr, edges_va, edges_te])

    n_hidden, num_layers, bsz = PAPER_HP[name]
    n_max_train = int(nodes_tr.max()) if len(nodes_tr) else 0
    peak = peak_cat_gb(bsz, n_max_train, n_hidden, num_layers)

    return (
        f"  {name:<10} train/val/test = {n_tr}/{n_va}/{n_te}\n"
        f"    nodes (min/p50/p95/p99/max, all splits): {fmt_dist(nodes_all)}\n"
        f"    edges (min/p50/p95/p99/max, all splits): {fmt_dist(edges_all)}\n"
        f"    train-only N: min={nodes_tr.min()} p95={int(np.percentile(nodes_tr, 95))} "
        f"max={n_max_train}\n"
        f"    paper HP: bsz={bsz} n_hidden={n_hidden} num_layers={num_layers} sigma={SIGMA_LENGTH}\n"
        f"    peak Powerful cat-tensor @ train N_max ≈ {peak:.2f} GB  "
        f"(fits T4 16GB: {peak < 14}, fits A100 40GB: {peak < 38})"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REPO_ROOT / "data"))
    p.add_argument("--datasets", nargs="+", default=list(PAPER_HP.keys()))
    args = p.parse_args()

    print(f"Dataset stats (root={args.root}, sigma_length={SIGMA_LENGTH})\n")
    for name in args.datasets:
        if name not in PAPER_HP:
            print(f"  [skip] {name}: not in PAPER_HP table")
            continue
        print(report(name, args.root))
        print()


if __name__ == "__main__":
    main()
