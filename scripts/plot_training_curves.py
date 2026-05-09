"""Plot test-time training curves for a paired baseline-vs-natgrad run.

Reads results/{dataset}/{baseline,natgrad}/metrics.jsonl and writes two-panel
PDF + PNG to paper/figs/training_curves_{dataset}.{pdf,png}.

Top panel: test CF-ACC vs epoch.
Bottom panel: test sparsity vs epoch (Tree-Cycle's ~3x sparsity inflation
under natgrad is a paper-worthy tension to surface).

Usage:
    python -m scripts.plot_training_curves --dataset Tree_Cycle
    python -m scripts.plot_training_curves --dataset ba3
"""

import argparse
import json
import os

import matplotlib.pyplot as plt


def load_test_records(path):
    """Read test-eval records from metrics.jsonl, deduped by epoch (last wins).

    Some runs restarted mid-training and re-appended to the same metrics.jsonl,
    producing duplicate-epoch records. Last-write-wins matches the
    `best_model.pth` selection (which always overwrites).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    by_epoch = {}
    for r in recs:
        if "test_acc" not in r:
            continue
        by_epoch[r["epoch"]] = r
    test_recs = [by_epoch[ep] for ep in sorted(by_epoch.keys())]
    eps = [r["epoch"] for r in test_recs]
    accs = [r["test_acc"] for r in test_recs]
    sparsities = [r["test_sparsity"] for r in test_recs]
    return eps, accs, sparsities


def rolling_mean(values, window):
    if len(values) < window:
        return list(values)
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def plot_pair(dataset, out_dir, smooth_window=5):
    base_path = f"results/{dataset}/baseline/metrics.jsonl"
    nat_path = f"results/{dataset}/natgrad/metrics.jsonl"

    eb, ab, sb = load_test_records(base_path)
    en, an, sn = load_test_records(nat_path)

    fig, (ax_acc, ax_sp) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

    base_color = "#2b6cb0"
    nat_color = "#c05621"

    def _draw(ax, x, y, color, label):
        ax.plot(x, y, color=color, lw=0.8, alpha=0.3)
        ax.plot(x, rolling_mean(y, smooth_window), color=color, lw=1.8, label=label)

    _draw(ax_acc, eb, ab, base_color, "Vanilla SGD baseline")
    _draw(ax_acc, en, an, nat_color, "Corrected natural gradient")
    # Peak markers
    ib = max(range(len(ab)), key=lambda i: ab[i])
    in_ = max(range(len(an)), key=lambda i: an[i])
    ax_acc.scatter([eb[ib]], [ab[ib]], s=80, marker="*", color=base_color,
                   edgecolor="black", linewidth=0.5, zorder=5,
                   label=f"Baseline best ({ab[ib]:.3f} @ ep {eb[ib]})")
    ax_acc.scatter([en[in_]], [an[in_]], s=80, marker="*", color=nat_color,
                   edgecolor="black", linewidth=0.5, zorder=5,
                   label=f"Natgrad best ({an[in_]:.3f} @ ep {en[in_]})")
    ax_acc.set_ylabel("Test CF-ACC")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(loc="lower right", frameon=False, fontsize=8)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(f"{dataset}: paired training trajectories (smoothed, window={smooth_window})")

    _draw(ax_sp, eb, sb, base_color, "Vanilla SGD baseline")
    _draw(ax_sp, en, sn, nat_color, "Corrected natural gradient")
    ax_sp.set_ylabel("Test sparsity")
    ax_sp.set_xlabel("Epoch")
    ax_sp.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"training_curves_{dataset}.pdf")
    png_path = os.path.join(out_dir, f"training_curves_{dataset}.png")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_b = max(zip(ab, eb))
    best_n = max(zip(an, en))
    print(f"Wrote {pdf_path} and {png_path}")
    print(f"  baseline: {len(eb)} eval points, best CF-ACC {best_b[0]:.4f} @ ep {best_b[1]}, final {ab[-1]:.4f}")
    print(f"  natgrad : {len(en)} eval points, best CF-ACC {best_n[0]:.4f} @ ep {best_n[1]}, final {an[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g. Tree_Cycle, ba3, NCI1, Tree_Grids")
    parser.add_argument("--out_dir", default="paper/figs")
    args = parser.parse_args()
    plot_pair(args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
