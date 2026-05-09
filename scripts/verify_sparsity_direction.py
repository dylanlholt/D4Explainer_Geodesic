"""Verify which direction the test_sparsity column points.

Loads ba3 baseline and natgrad checkpoints, runs both on the test set,
and reports for each:
  - modif_r           = |GT - pred|.sum() / GT.sum()           (the logged column)
  - mean #pred edges
  - mean #GT edges
  - retention rate    = (pred AND GT).sum() / GT.sum()
  - addition rate     = (pred AND NOT GT).sum() / GT.sum()
  - true sparsity (D4Explainer convention) = retention rate

If natgrad has FEWER mean predicted edges => natgrad is genuinely sparser.
If natgrad has MORE mean predicted edges  => natgrad is genuinely denser
                                              (and the memory labels were inverted).
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

# Shim: checkpoints were saved by a newer numpy that exposes numpy._core.
# Older numpy in this env only has numpy.core; alias them so torch.load
# can resolve pickled references to numpy._core.*.
import sys as _sys
if not hasattr(np, "_core"):
    import numpy.core as _npcore
    _sys.modules["numpy._core"] = _npcore
    _sys.modules["numpy._core.multiarray"] = _npcore.multiarray
    _sys.modules["numpy._core.numeric"] = _npcore.numeric

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainers.diffusion.graph_utils import (
    gen_list_of_data_single,
    generate_mask,
    graph2tensor,
)
from explainers.diffusion.pgnn import Powerful
from utils.dataset import get_datasets


def load_args(config_path, device):
    with open(config_path) as f:
        cfg = json.load(f)
    cfg["device"] = device
    cfg["noise_list"] = None
    return SimpleNamespace(**cfg)


@torch.no_grad()
def eval_arm(arm_dir, dataset_name, device, max_batches=None):
    args = load_args(os.path.join(arm_dir, "config.json"), device)
    model = Powerful(args).to(device)
    sd = torch.load(os.path.join(arm_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval()

    _, _, test_dataset = get_datasets(name=dataset_name)
    loader = DataLoader(dataset=test_dataset, batch_size=args.test_batchsize, shuffle=False)

    sums = {
        "modif_r": [],
        "n_pred": [],
        "n_gt": [],
        "n_intersect": [],
        "n_added": [],
    }

    for bi, graph in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        if graph.is_directed():
            graph.edge_index = to_undirected(graph.edge_index)
        graph.to(device)
        test_adj_b, test_x_b = graph2tensor(graph, device=device)
        test_node_flag_b = test_adj_b.sum(-1).gt(1e-5).to(dtype=torch.float32)

        sigma_list = list(np.random.uniform(low=args.prob_low, high=args.prob_high, size=args.sigma_length))
        (test_x_b, _, test_node_flag_sigma, test_noise_adj_b, _) = gen_list_of_data_single(
            test_x_b, test_adj_b, test_node_flag_b, sigma_list, args
        )
        test_noise_adj_b_chunked = test_noise_adj_b.chunk(len(sigma_list), dim=0)
        test_x_b_chunked = test_x_b.chunk(len(sigma_list), dim=0)
        test_node_flag_sigma = test_node_flag_sigma.chunk(len(sigma_list), dim=0)

        scores = []
        last_mask = None
        for i, sigma in enumerate(sigma_list):
            mask = generate_mask(test_node_flag_sigma[i])
            s = model(
                A=test_noise_adj_b_chunked[i].to(device),
                node_features=test_x_b_chunked[i].to(device),
                mask=mask.to(device),
                noiselevel=sigma,
            )
            scores.append(s.squeeze(-1))
            last_mask = mask.to(device)

        score_tensor = torch.stack(scores, dim=0)  # [S, B, N, N]
        score_mean = score_tensor.mean(dim=0)
        pred_adj = (torch.sigmoid(score_mean) > args.threshold).float() * last_mask
        gt_adj = test_adj_b * last_mask

        diff = (gt_adj - pred_adj).abs()
        n_gt = gt_adj.sum().item()
        n_pred = pred_adj.sum().item()
        n_intersect = (gt_adj * pred_adj).sum().item()
        n_added = ((1 - gt_adj) * pred_adj * last_mask).sum().item()

        sums["modif_r"].append(diff.sum().item() / max(n_gt, 1))
        sums["n_pred"].append(n_pred)
        sums["n_gt"].append(n_gt)
        sums["n_intersect"].append(n_intersect)
        sums["n_added"].append(n_added)

    return {k: np.mean(v) for k, v in sums.items()}, len(sums["modif_r"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ba3")
    ap.add_argument("--baseline_dir", default="results/ba3/baseline")
    ap.add_argument("--natgrad_dir", default="results/ba3/natgrad")
    ap.add_argument("--max_batches", type=int, default=None,
                    help="Limit batches for a quick check (None = full test set)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"device={device}")

    print(f"\n=== {args.dataset} :: BASELINE ===")
    base, nb = eval_arm(args.baseline_dir, args.dataset, device, args.max_batches)
    print(f"batches={nb}")
    for k, v in base.items():
        print(f"  {k:>12s} = {v:.4f}")

    print(f"\n=== {args.dataset} :: NATGRAD ===")
    nat, nn = eval_arm(args.natgrad_dir, args.dataset, device, args.max_batches)
    print(f"batches={nn}")
    for k, v in nat.items():
        print(f"  {k:>12s} = {v:.4f}")

    print("\n=== INTERPRETATION ===")
    base_ret = base["n_intersect"] / max(base["n_gt"], 1e-9)
    nat_ret = nat["n_intersect"] / max(nat["n_gt"], 1e-9)
    print(f"  baseline retention = {base_ret:.4f}  (#pred={base['n_pred']:.1f},  #gt={base['n_gt']:.1f})")
    print(f"  natgrad  retention = {nat_ret:.4f}  (#pred={nat['n_pred']:.1f},  #gt={nat['n_gt']:.1f})")
    print(f"  baseline modif_r   = {base['modif_r']:.4f}")
    print(f"  natgrad  modif_r   = {nat['modif_r']:.4f}")

    if nat["n_pred"] < base["n_pred"]:
        print("\n  >>> NATGRAD predicts FEWER edges => natgrad is GENUINELY SPARSER.")
    else:
        print("\n  >>> NATGRAD predicts MORE/EQUAL edges => natgrad is GENUINELY DENSER.")

    print("\n  D4Explainer convention (CONTEXT.md): sparsity = fraction of edges retained,")
    print("  lower-is-better. Under that convention, the sparsity numbers are:")
    print(f"    baseline true_sparsity = retention = {base_ret:.4f}")
    print(f"    natgrad  true_sparsity = retention = {nat_ret:.4f}")


if __name__ == "__main__":
    main()
