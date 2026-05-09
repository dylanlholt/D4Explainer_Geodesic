"""BA-3Motif applied gallery: 3 instances, 2x2 grid each.

For each chosen test instance, render four panels:
  (a) input graph (all edges, faded)
  (b) ground-truth motif (motif edges highlighted)
  (c) baseline explanation (retained edges highlighted)
  (d) corrected natgrad explanation (retained edges highlighted)

Usage:
    python -m scripts.plot_ba3_gallery --instances 0 1 2

Defaults pick one instance per class label (cycle / house / grid).
Requires both results/ba3/baseline/best_model.pth and
results/ba3/natgrad/best_model.pth to exist.
"""

import argparse
import os

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Batch

from gnns import *  # noqa: F401,F403 — register GNN classes on this module so torch.load can unpickle GNN checkpoints saved by the gnns/*.py training scripts (which pickle the class against __main__).

from scripts._explain_helpers import (
    assert_checkpoint_exists,
    build_explainer,
    load_args_from_run,
    load_test_dataset,
)


def pick_default_instances(dataset, n=3):
    """Pick n instances spanning class labels for visual variety."""
    seen = {}
    for i in range(len(dataset)):
        y = int(dataset[i].y.item())
        if y not in seen:
            seen[y] = i
        if len(seen) == n:
            break
    return [seen[k] for k in sorted(seen.keys())][:n]


def explain_one(explainer, args, data):
    batch = Batch.from_data_list([data]).to(args.device)
    with torch.no_grad():
        graph_sub, y_ori, y_exp, modif_r = explainer.explain_evaluation(args, batch)
    edges = graph_sub.edge_index.cpu().numpy()
    return edges, int(y_exp.item()), float(modif_r.item() if torch.is_tensor(modif_r) else modif_r)


def edges_to_set(edge_index):
    """Return a set of frozenset({u,v}) — undirected edges."""
    return {frozenset((int(u), int(v))) for u, v in zip(edge_index[0], edge_index[1]) if u != v}


def draw_graph(ax, data, highlight_edges, title, all_edges_alpha=0.15):
    G = nx.Graph()
    n_nodes = data.x.size(0)
    G.add_nodes_from(range(n_nodes))
    full_edges = edges_to_set(data.edge_index.cpu().numpy())
    G.add_edges_from([tuple(e) for e in full_edges])
    pos = {i: (float(data.pos[i][0]), float(data.pos[i][1])) for i in range(n_nodes)}
    other = full_edges - highlight_edges
    nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in other],
                           alpha=all_edges_alpha, edge_color="grey", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in highlight_edges],
                           width=2.0, edge_color="#c05621", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color="#2b6cb0", ax=ax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ba3")
    parser.add_argument("--instances", type=int, nargs="*", default=None,
                        help="Instance indices in test split. Default: one per class.")
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--natgrad_run", default="natgrad")
    parser.add_argument("--out", default="paper/figs/ba3_gallery.pdf")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()

    test = load_test_dataset(args_cli.dataset)
    instances = args_cli.instances or pick_default_instances(test, n=3)
    print(f"Picked instances: {instances}")

    base_dir = f"results/{args_cli.dataset}/{args_cli.baseline_run}"
    nat_dir = f"results/{args_cli.dataset}/{args_cli.natgrad_run}"
    assert_checkpoint_exists(base_dir)
    assert_checkpoint_exists(nat_dir)

    device = torch.device(args_cli.device)
    explainer = build_explainer(device, args_cli.dataset)

    base_args = load_args_from_run(base_dir, device)
    nat_args = load_args_from_run(nat_dir, device)

    n = len(instances)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.6 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    class_names = {0: "cycle", 1: "house", 2: "grid"}

    for row, idx in enumerate(instances):
        data = test[idx]
        full_edges = edges_to_set(data.edge_index.cpu().numpy())
        gt_mask = data.ground_truth_mask
        gt_pairs = [(int(u), int(v)) for (u, v), m in zip(data.edge_index.cpu().numpy().T, gt_mask) if m]
        gt_edges = {frozenset(e) for e in gt_pairs}

        base_edges_idx, base_pred, base_modif = explain_one(explainer, base_args, data)
        nat_edges_idx, nat_pred, nat_modif = explain_one(explainer, nat_args, data)
        base_edges = edges_to_set(base_edges_idx) & full_edges
        nat_edges = edges_to_set(nat_edges_idx) & full_edges

        cls = class_names.get(int(data.y.item()), str(int(data.y.item())))
        draw_graph(axes[row, 0], data, set(),
                   f"#{idx} input  (class={cls})")
        draw_graph(axes[row, 1], data, gt_edges,
                   f"ground-truth motif")
        draw_graph(axes[row, 2], data, base_edges,
                   f"baseline  (sparsity {base_modif:.2f})")
        draw_graph(axes[row, 3], data, nat_edges,
                   f"natgrad   (sparsity {nat_modif:.2f})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    plt.savefig(args_cli.out, bbox_inches="tight")
    png = args_cli.out.replace(".pdf", ".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args_cli.out} and {png}")


if __name__ == "__main__":
    main()
