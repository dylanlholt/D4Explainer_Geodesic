"""Fisher information visualisation: 1D curve + 2D heatmap with cap.

Schematic figure for §4.2. Visualises the diagonal Bernoulli FIM
$I(\\theta) = 1/(\\theta(1-\\theta))$ in two complementary forms:

(a) 1D curve over $\\theta \\in (0,1)$, with the rescaling cap (default
    100) drawn as a horizontal line. Highlights where the cap clamps the
    rescaling factor (very near $\\theta \\in \\{0, 1\\}$).

(b) 2D heatmap of the average per-edge factor
    $\\tfrac12 (I(\\theta_1) + I(\\theta_2))$ on $(0,1)^2$, with the cap
    contour overlaid. Shows where in the explanation manifold the
    rescaling is most aggressive.

Pure schematic; no model checkpoints required.

Usage:
    python -m scripts.plot_fisher_information --cap 100
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=float, default=100.0,
                        help="Rescaling factor cap used in implementation.")
    parser.add_argument("--out", default="paper/figs/fisher_information.pdf")
    args = parser.parse_args()

    cap = args.cap

    fig, (ax_1d, ax_2d) = plt.subplots(1, 2, figsize=(11, 4.4))

    # === Left: 1D curve ===
    theta = np.linspace(0.001, 0.999, 1000)
    fim = 1.0 / (theta * (1.0 - theta))

    ax_1d.plot(theta, fim, color="#c05621", lw=2.0,
               label=r"$I(\theta) = 1 / (\theta(1-\theta))$")
    ax_1d.axhline(cap, color="#2b6cb0", lw=1.5, linestyle="--",
                  label=f"rescaling cap = {cap:.0f}")
    ax_1d.fill_between(theta, fim, cap, where=(fim > cap),
                       color="#2b6cb0", alpha=0.15,
                       label="cap-active region")
    ax_1d.set_xlabel(r"Edge probability $\theta$")
    ax_1d.set_ylabel(r"Diagonal Fisher information / rescaling factor")
    ax_1d.set_yscale("log")
    ax_1d.set_xlim(0, 1)
    ax_1d.set_ylim(3, 5e3)
    ax_1d.grid(True, alpha=0.3, which="both")
    ax_1d.legend(loc="upper center", frameon=False, fontsize=9)
    ax_1d.set_title(r"(a) 1D Fisher information $I(\theta)$ with cap", fontsize=10)

    # Annotate the floor: I(0.5) = 4
    ax_1d.scatter([0.5], [4.0], s=40, color="black", zorder=5)
    ax_1d.annotate(r"$I(0.5) = 4$ (floor)", (0.5, 4.0),
                   textcoords="offset points", xytext=(10, 4), fontsize=8)

    # Annotate where cap kicks in
    theta_cap = (1.0 - np.sqrt(1.0 - 4.0 / cap)) / 2.0
    ax_1d.scatter([theta_cap], [cap], s=40, color="#2b6cb0", zorder=5)
    ax_1d.annotate(f"cap active at\n" + r"$\theta < {:.3f}$, $\theta > {:.3f}$".format(theta_cap, 1 - theta_cap),
                   (theta_cap, cap), textcoords="offset points", xytext=(8, 8), fontsize=8)

    # === Right: 2D heatmap ===
    grid = np.linspace(0.005, 0.995, 200)
    T1, T2 = np.meshgrid(grid, grid)
    I1 = 1.0 / (T1 * (1 - T1))
    I2 = 1.0 / (T2 * (1 - T2))
    avg_factor = 0.5 * (np.minimum(I1, cap) + np.minimum(I2, cap))

    # Plot raw average factor (no cap) under the heatmap to show the
    # divergence shape; overlay the cap contour.
    avg_uncapped = 0.5 * (I1 + I2)
    log_avg = np.log10(avg_uncapped)
    pcm = ax_2d.pcolormesh(T1, T2, log_avg, cmap="OrRd", shading="auto",
                           vmin=np.log10(4.0), vmax=np.log10(1e3))
    cs = ax_2d.contour(T1, T2, np.maximum(I1, I2), levels=[cap],
                       colors=["#2b6cb0"], linewidths=2.0, linestyles=["--"])
    ax_2d.clabel(cs, inline=True, fmt={cap: f"cap = {cap:.0f}"}, fontsize=9)

    ax_2d.set_xlabel(r"$\theta_1$ (edge 1 probability)")
    ax_2d.set_ylabel(r"$\theta_2$ (edge 2 probability)")
    ax_2d.set_aspect("equal")
    ax_2d.set_title(r"(b) Average per-edge rescaling factor on $\mathcal{M}_G$", fontsize=10)

    cbar = fig.colorbar(pcm, ax=ax_2d, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\log_{10} \frac{1}{2}[I(\theta_1) + I(\theta_2)]$",
                   fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    png = args.out.replace(".pdf", ".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
