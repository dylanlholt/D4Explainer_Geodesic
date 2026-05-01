"""Natural gradient utilities for the Bernoulli-product explanation manifold.

Setup. The Powerful denoiser emits a per-edge logit ``ℓ_ij``; the Bernoulli edge
probability is ``θ_ij = σ(ℓ_ij)``. The Bernoulli-product manifold ``M_G`` carries
the diagonal Fisher-Rao metric ``g_θ = 1 / (θ(1-θ))``.

Natural gradient on θ-space is ``g_θ^{-1} ∂L/∂θ = θ(1-θ) ∂L/∂θ``. The optimizer
updates network parameters ``w`` via logit-space SGD, and the chain rule gives
``∂L/∂ℓ = θ(1-θ) ∂L/∂θ``. Substituting, the parameter update that realises the
θ-space natural-gradient direction is

    Δw ∝ (1/(θ(1-θ))) · ∂L/∂ℓ · ∂ℓ/∂w,

so the correct backward-hook rescaling on the *logit* gradient is the
**inverse** factor ``1/(θ(1-θ))`` — not ``θ(1-θ)``.

This factor diverges at θ ∈ {0, 1}; we clamp θ via ``epsilon`` and additionally
cap the rescaling at ``cap`` to bound update size. Bernoulli factorization keeps
the rescaling diagonal — O(|E|) per edge.
"""

import torch


def compute_fim_diagonal(logit, mask, epsilon=1e-6):
    """Diagonal Fisher information ``I(θ)_ij = 1/(θ(1-θ))`` from a logit tensor.

    Args:
        logit: [bsz, N, N, 1] or [bsz, N, N] model output (per-edge logit).
        mask:  [bsz, N, N] valid-edge indicator.
        epsilon: clamp θ ∈ [ε, 1-ε] to avoid singularity at θ ∈ {0, 1}.

    Returns:
        [bsz, N, N] diagonal FIM, zeroed outside the mask.
    """
    if logit.dim() == 4:
        logit = logit.squeeze(-1)
    theta = torch.sigmoid(logit).clamp(epsilon, 1.0 - epsilon)
    fim = 1.0 / (theta * (1.0 - theta))
    return fim * mask


def register_natural_gradient_hook(score_batch, epsilon=1e-6, cap=100.0):
    """Register a backward hook that rescales ``∂L/∂ℓ`` by ``1/(θ(1-θ))``.

    Call immediately after the model forward pass, before backward. The hook
    fires when gradient flows back through ``score_batch`` during
    ``loss.backward()``.

    Args:
        score_batch: [bsz, N, N, 1] model output (logit) requiring grad.
        epsilon: clamp θ ∈ [ε, 1-ε] before inverting the FIM.
        cap: upper bound on the rescaling factor; bounds update size at extreme
            logits where ``1/(θ(1-θ))`` would otherwise diverge.
    """
    if not score_batch.requires_grad:
        return
    theta = torch.sigmoid(score_batch.detach()).clamp(epsilon, 1.0 - epsilon)
    rescale = (1.0 / (theta * (1.0 - theta))).clamp(max=cap)

    def _hook(grad):
        return grad * rescale

    score_batch.register_hook(_hook)
