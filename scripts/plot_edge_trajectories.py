"""Edge-probability trajectory plot: theta_ij vs noise level sigma.

For one chosen test instance, runs both checkpoints (baseline and corrected
natgrad) at a sweep of sigma values and plots theta_ij = sigmoid(model logit)
for K representative edges. The picked edges are a mix of "decided"
(baseline final theta near 0/1) and "uncertain" (theta near 0.5), which is
the regime where the FR rescaling factor 1/(theta(1-theta)) differs most
between methods. This visualizes the §4.2 claim that natgrad amplifies
updates at decided edges.

Note: this is *not* a true T-step reverse-diffusion trajectory. The
underlying inference path in this codebase is one-shot multi-sigma
averaging (`explain_evaluation`); the plot here is the per-sigma sensitivity
of each method, which captures the geometric content without re-implementing
reverse diffusion.

Usage:
    python -m scripts.plot_edge_trajectories --dataset Tree_Cycle --instance 0
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

from gnns import *  # noqa: F401,F403 — register GNN classes on this module so torch.load can unpickle GNN checkpoints saved by the gnns/*.py training scripts.

from explainers.diffusion.graph_utils import generate_mask, graph2tensor
from explainers.diffusion.pgnn import Powerful

from scripts._explain_helpers import (
    assert_checkpoint_exists,
    load_args_from_run,
    load_test_dataset,
)


def load_model(args, run_dir):
    model = Powerful(args).to(args.device)
    state = torch.load(os.path.join(run_dir, "best_model.pth"), weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def per_sigma_logits(model, args, data, sigmas):
    """Return [len(sigmas), N, N] logits — model output at each sigma on the noisy adj."""
    batch = Batch.from_data_list([data]).to(args.device)
    A, X = graph2tensor(batch, device=args.device)
    node_flag = A.sum(-1).gt(1e-5).to(dtype=torch.float32)
    mask = generate_mask(node_flag).to(args.device)
    out = []
    for s in sigmas:
        score = model(A=A, node_features=X, mask=mask, noiselevel=float(s))
        out.append(score.squeeze(0).squeeze(-1).cpu().numpy())  # [N, N]
    return np.stack(out, axis=0)  # [len(sigmas), N, N]


def pick_edges(logits_baseline, mask, n_decided=4, n_uncertain=4):
    """Pick edges based on baseline final theta: half decided, half uncertain."""
    theta_final = 1.0 / (1.0 + np.exp(-logits_baseline[0]))  # sigma 0 ≈ noise-free
    theta_final = theta_final * mask
    # Use upper triangle to avoid double-counting undirected edges
    iu = np.triu_indices_from(theta_final, k=1)
    flat_theta = theta_final[iu]
    valid = mask[iu] > 0
    flat_theta = flat_theta[valid]
    iu_i, iu_j = iu[0][valid], iu[1][valid]
    # decided: |theta - 0.5| largest; uncertain: |theta - 0.5| smallest
    distance = np.abs(flat_theta - 0.5)
    decided_idx = np.argsort(-distance)[:n_decided]
    uncertain_idx = np.argsort(distance)[:n_uncertain]
    edges = []
    for i in decided_idx:
        edges.append(("decided", int(iu_i[i]), int(iu_j[i]), float(flat_theta[i])))
    for i in uncertain_idx:
        edges.append(("uncertain", int(iu_i[i]), int(iu_j[i]), float(flat_theta[i])))
    return edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--baseline_run", default="baseline")
    parser.add_argument("--natgrad_run", default="natgrad")
    parser.add_argument("--n_sigma", type=int, default=20)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args_cli = parser.parse_args()

    out = args_cli.out or f"paper/figs/edge_trajectories_{args_cli.dataset}.pdf"

    test = load_test_dataset(args_cli.dataset)
    data = test[args_cli.instance]

    base_dir = f"results/{args_cli.dataset}/{args_cli.baseline_run}"
    nat_dir = f"results/{args_cli.dataset}/{args_cli.natgrad_run}"
    assert_checkpoint_exists(base_dir)
    assert_checkpoint_exists(nat_dir)

    device = torch.device(args_cli.device)
    base_args = load_args_from_run(base_dir, device)
    nat_args = load_args_from_run(nat_dir, device)

    sigmas = np.linspace(base_args.prob_low, base_args.prob_high, args_cli.n_sigma)

    base_model = load_model(base_args, base_dir)
    nat_model = load_model(nat_args, nat_dir)

    base_logits = per_sigma_logits(base_model, base_args, data, sigmas)
    nat_logits = per_sigma_logits(nat_model, nat_args, data, sigmas)

    # Build mask once
    batch = Batch.from_data_list([data]).to(device)
    A, _ = graph2tensor(batch, device=device)
    node_flag = A.sum(-1).gt(1e-5).to(dtype=torch.float32)
    mask = generate_mask(node_flag).squeeze(0).cpu().numpy()

    # Pick representative edges from baseline's lowest-sigma (cleanest) prediction
    edges = pick_edges(base_logits, mask)
    print(f"Picked {len(edges)} edges:")
    for kind, u, v, theta in edges:
        print(f"  {kind:10s} ({u:3d},{v:3d}) baseline final theta={theta:.3f}")

    fig, (ax_b, ax_n) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    decided_color = "#c05621"
    uncertain_color = "#2b6cb0"

    for kind, u, v, _ in edges:
        color = decided_color if kind == "decided" else uncertain_color
        theta_b = 1.0 / (1.0 + np.exp(-base_logits[:, u, v]))
        theta_n = 1.0 / (1.0 + np.exp(-nat_logits[:, u, v]))
        ax_b.plot(sigmas, theta_b, color=color, alpha=0.8, lw=1.2)
        ax_n.plot(sigmas, theta_n, color=color, alpha=0.8, lw=1.2)

    for ax, title in [(ax_b, "Vanilla SGD baseline"), (ax_n, "Corrected natural gradient")]:
        ax.set_xlabel(r"Noise level $\sigma$")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="grey", alpha=0.3, lw=0.7)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
    ax_b.set_ylabel(r"Edge probability $\theta_{ij} = \sigma(\ell_{ij})$")

    # Custom legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=decided_color, lw=1.5,
               label=r"Decided edges ($|\theta - 0.5|$ large)"),
        Line2D([0], [0], color=uncertain_color, lw=1.5,
               label=r"Uncertain edges ($\theta \approx 0.5$)"),
    ]
    ax_b.legend(handles=handles, loc="best", frameon=False, fontsize=9)

    plt.suptitle(f"{args_cli.dataset} instance #{args_cli.instance}: per-edge "
                 r"$\theta$ vs.\ $\sigma$",
                 fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    png = out.replace(".pdf", ".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} and {png}")


if __name__ == "__main__":
    main()
