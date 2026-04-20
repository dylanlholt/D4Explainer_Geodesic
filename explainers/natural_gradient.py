"""Natural gradient utilities for the Bernoulli-product explanation manifold.

Each model output ``score`` is interpreted as an edge probability ``θ_ij`` on
the product manifold ``M_G = [0,1]^|E|``. Under independent Bernoullis the
Fisher-Rao metric is diagonal with ``I(θ)_ij = 1 / (θ_ij (1 - θ_ij))``, so the
natural-gradient rescaling ``I(θ)^{-1} = θ(1-θ)`` is element-wise — O(|E|).

The hook rescales ``dL/d(score)`` in place during ``backward()``; the chain rule
then propagates this natural-gradient-adjusted signal through the network, so
``optimizer.step()`` follows a geodesic direction on M_G.
"""

import torch


def compute_fim_diagonal(score, mask, epsilon=1e-6):
    """Diagonal Fisher information I(θ)_ij for a single score tensor.

    Args:
        score: [bsz, N, N, 1] or [bsz, N, N] model output (edge probabilities).
        mask:  [bsz, N, N] valid-edge indicator.
        epsilon: clamp to avoid singularity at θ ∈ {0, 1}.

    Returns:
        [bsz, N, N] diagonal FIM, zeroed outside the mask.
    """
    if score.dim() == 4:
        score = score.squeeze(-1)
    theta = score.clamp(epsilon, 1.0 - epsilon)
    fim = 1.0 / (theta * (1.0 - theta))
    return fim * mask


def register_natural_gradient_hook(score_batch, epsilon=1e-6):
    """Register a backward hook that applies the diagonal I(θ)^{-1} rescaling.

    Call immediately after the model forward pass, before backward. The hook
    fires when gradient flows back through ``score_batch`` during
    ``loss.backward()``.

    Args:
        score_batch: [bsz, N, N, 1] model output requiring grad.
        epsilon: boundary clamp for θ.
    """
    if not score_batch.requires_grad:
        return
    theta = score_batch.detach().clamp(epsilon, 1.0 - epsilon)
    inv_fim = theta * (1.0 - theta)

    def _hook(grad):
        return grad * inv_fim

    score_batch.register_hook(_hook)
