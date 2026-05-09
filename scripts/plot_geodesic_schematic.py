"""2D Bernoulli product manifold: Fisher-Rao geodesic vs Euclidean line.

Schematic figure for §3. Shows that on the explanation manifold $\\MG$ with
the Fisher-Rao metric $g = \\mathrm{diag}(1/(\\theta(1-\\theta)))$, the
Riemannian geodesic between two points curves away from the boundary of
$(0,1)^2$, while the Euclidean straight line cuts across uniformly.

The FR geodesic is computed via the standard isometry to flat space:
$\\phi_i = 2\\arcsin(\\sqrt{\\theta_i})$ flattens the metric, geodesics
become straight lines in $\\phi$-space, and $\\theta_i = \\sin^2(\\phi_i/2)$
maps back. (See e.g. Amari 2016 §2.6 for the Bernoulli manifold geometry.)

Pure schematic: no model checkpoints required.

Usage:
    python -m scripts.plot_geodesic_schematic
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def to_phi(theta):
    return 2.0 * np.arcsin(np.sqrt(theta))


def from_phi(phi):
    return np.sin(phi / 2.0) ** 2


def fr_geodesic(theta_a, theta_b, n=200):
    """Geodesic from theta_a to theta_b on Bernoulli^2 with FR metric."""
    phi_a = to_phi(np.array(theta_a))
    phi_b = to_phi(np.array(theta_b))
    ts = np.linspace(0.0, 1.0, n)
    phi_t = (1.0 - ts[:, None]) * phi_a + ts[:, None] * phi_b
    return from_phi(phi_t)


def euclidean_line(theta_a, theta_b, n=200):
    a = np.array(theta_a)
    b = np.array(theta_b)
    ts = np.linspace(0.0, 1.0, n)
    return (1.0 - ts[:, None]) * a + ts[:, None] * b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="paper/figs/geodesic_schematic.pdf")
    args = parser.parse_args()

    # Two endpoint pairs chosen so the FR geodesic visibly diverges from
    # the Euclidean chord. Symmetric pairs (e.g.\ diagonal) coincide on
    # this manifold and don't illustrate the curvature, so we use
    # asymmetric configurations that pass through the high-curvature
    # boundary regions.
    pairs = [
        ((0.01, 0.20), (0.50, 0.99), "Asymmetric corner-to-edge"),
        ((0.10, 0.10), (0.95, 0.50), "Decided-edge to mid-uncertain"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))

    for ax, ((a, b), title) in zip(axes, [(p[:2], p[2]) for p in pairs]):
        # Background: Fisher info heatmap as visual cue for "where curvature lives"
        thetas = np.linspace(0.005, 0.995, 200)
        T1, T2 = np.meshgrid(thetas, thetas)
        # Volume element sqrt(det g) = 1 / sqrt(theta1 (1-theta1) theta2 (1-theta2))
        vol = 1.0 / np.sqrt(T1 * (1 - T1) * T2 * (1 - T2))
        log_vol = np.log10(vol)
        im = ax.pcolormesh(T1, T2, log_vol, cmap="Blues", alpha=0.4, shading="auto",
                           vmin=log_vol.min(), vmax=log_vol.max())

        eu = euclidean_line(a, b)
        fr = fr_geodesic(a, b)

        ax.plot(eu[:, 0], eu[:, 1], color="#2b6cb0", lw=2.0,
                label="Euclidean (vanilla SGD direction)")
        ax.plot(fr[:, 0], fr[:, 1], color="#c05621", lw=2.0,
                label="Fisher-Rao geodesic (natural gradient)")
        ax.scatter([a[0], b[0]], [a[1], b[1]], s=60, color="black", zorder=5)
        ax.annotate("A", a, textcoords="offset points", xytext=(-12, -10), fontsize=10)
        ax.annotate("B", b, textcoords="offset points", xytext=(8, 4), fontsize=10)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\theta_1$ (edge 1 probability)")
        ax.set_ylabel(r"$\theta_2$ (edge 2 probability)")
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right" if a[1] > 0.3 else "upper right",
                  frameon=False, fontsize=8)

    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label(r"$\log_{10}\sqrt{\det g_{\mathrm{FR}}(\theta)}$  (volume distortion)", fontsize=9)

    fig.suptitle(r"Bernoulli$^2$ explanation manifold: Euclidean line vs.\ Fisher-Rao geodesic",
                 fontsize=11)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    png = args.out.replace(".pdf", ".png")
    plt.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out} and {png}")


if __name__ == "__main__":
    main()
