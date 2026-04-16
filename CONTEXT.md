# CONTEXT.md
## Project: Geodesic Attribution on the Graph Explanation Manifold
### *Navigating D4Explainer's Counterfactual Explanation Space via Fisher-Rao Geodesic Flow*

---

## 1. Project Purpose

This project extends **D4Explainer** (Chen et al., NeurIPS 2023), a diffusion-based GNN explainability framework, by replacing its Euclidean denoising objective with **natural gradient descent** — enforcing that each reverse diffusion step follows the geometry of the graph distribution manifold rather than treating parameter space as flat.

**The one-line description:**
> *Navigating D4Explainer's counterfactual explanation space as geodesic flow on the graph distribution manifold under the Fisher-Rao metric via natural gradient descent.*

### Three Contributions
1. **Geometric reframing:** D4Explainer's explanation space is a statistical manifold $\mathcal{M}_G$; the correct distance on it is the Fisher-Rao metric, not Euclidean distance on adjacency matrices.
2. **Algorithmic replacement:** Natural gradient descent replaces the Euclidean optimizer; each denoising step follows the geodesic on $\mathcal{M}_G$.
3. **Interpretive payoff:** Per-edge attribution scores $\phi_{ij}$ arise as martingale increments along the geodesic — a completeness-satisfying alternative to Shapley value sampling.

### What Is NOT Changed
- The loss function: `loss = loss_dist + alpha_cf * loss_cf` is kept as-is.
- The model architecture and forward pass.
- The evaluation framework (D4Explainer benchmark: BA-Shapes, BA-Community, Tree-Cycles, Tree-Grid).

---

## 2. Mathematical Background

### 2.1 The Graph Explanation Manifold $\mathcal{M}_G$

Each candidate explanation is parameterized as $\theta \in [0,1]^{|E|}$, where $\theta_{ij} = P(\text{edge } (i,j) \in G')$.

Each $\theta$ defines a distribution over graphs via independent Bernoullis:
$$p_\theta(A) = \prod_{(i,j)} \theta_{ij}^{A_{ij}}(1-\theta_{ij})^{1-A_{ij}}$$

This makes $\mathcal{M}_G$ a **product statistical manifold** with a diagonal Fisher-Rao metric:
$$g_{ij}(\theta) = \frac{1}{\theta_{ij}(1 - \theta_{ij})}$$

This is the key fact that makes the method computationally tractable: $\mathcal{I}(\theta)^{-1}$ is $O(|E|)$ to compute, not $O(|E|^2)$.

### 2.2 The Natural Gradient Update

Standard gradient descent treats parameter space as Euclidean:
$$\theta^{(t-1)} = \theta^{(t)} - \eta \nabla_\theta \mathcal{L}$$

Natural gradient descent corrects for the curvature of $\mathcal{M}_G$:
$$\theta^{(t-1)} = \theta^{(t)} - \eta \cdot \mathcal{I}(\theta^{(t)})^{-1} \nabla_\theta \mathcal{L}$$

For the diagonal Bernoulli manifold this simplifies to element-wise rescaling:
$$\tilde{\nabla}_\theta \mathcal{L}_{ij} = \theta_{ij}(1 - \theta_{ij}) \cdot \nabla_\theta \mathcal{L}_{ij}$$

### 2.3 Attribution via Martingale Increments

The geodesic path $\{\theta^{(t)}\}_{t=T}^{0}$ induces a martingale:
$$M_t = \mathbb{E}[f(e^*_\text{geo}) \mid \theta^{(t)}]$$

Per-edge attribution scores are defined as:
$$\phi_{ij} = \sum_t \Delta M_t \cdot \frac{|\theta^{(t+1)}_{ij} - \theta^{(t)}_{ij}|}{\sum_{(a,b)} |\theta^{(t+1)}_{ab} - \theta^{(t)}_{ab}|}$$

**Completeness property (Shapley efficiency analog):**
$$\sum_{(i,j) \in E} \phi_{ij} = f(e^*_\text{geo}) - f(G)$$

---

## 3. Key Literature

| Paper | Role in Project |
|---|---|
| Chen et al. (2023) — **D4Explainer** | Framework being extended; evaluation protocol |
| Amari (2016) — *Information Geometry* | Fisher-Rao metric; natural gradient; geodesic flow |
| Song et al. (2021) — Score-based SDEs | Reverse-time SDE; score = Fisher score of $p_t$ |
| Jo et al. (2022) — **GDSS** | Continuous-time graph diffusion; parent of D4Explainer |
| Øksendal (2010) — *SDEs* | Itô diffusion theoretical foundation |
| Doshi-Velez & Kim (2017) | Functionally-grounded interpretability; motivates in-distribution constraint |
| Ly et al. (2017) | Fisher information as operational metric; Cramér-Rao |
| Coifman & Lafon (2006) — Diffusion Maps | Spectral geometry of $\mathcal{M}_G$; future extension |

### Full References
- Amari, S. (2016). *Information geometry and its applications*. Springer.
- Chen, J., Wu, S., Gupta, A., & Ying, R. (2023). D4Explainer: In-distribution GNN explanations via discrete denoising diffusion. *NeurIPS 2023*. arXiv:2310.19321.
- Coifman, R. R., & Lafon, S. (2006). Diffusion maps. *Applied and Computational Harmonic Analysis, 21*(1), 5–30.
- Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. arXiv:1702.08608.
- Jo, J., Lee, S., & Hwang, S. J. (2022). Score-based generative modeling of graphs via the system of SDEs. *ICML 2022*.
- Ly, A. et al. (2017). A tutorial on Fisher information. *Journal of Mathematical Psychology, 80*, 40–55.
- Øksendal, B. (2010). *Stochastic differential equations* (6th ed.). Springer.
- Song, Y. et al. (2021). Score-based generative modeling through SDEs. *ICLR 2021*.

---

## 4. Paper Outline (Structured)

### §1 Introduction
- GNNs lack intrinsic interpretability; counterfactual explanations offer a principled alternative
- Prior methods search Euclidean edge-mask space → OOD artifacts
- D4Explainer enforces in-distribution constraint via diffusion — key advance
- **Gap:** Denoising objective is Euclidean; ignores curvature of the manifold it constructs
- Three contributions stated above

### §2 Background
- GNN explainability taxonomy (gradient, perturbation, counterfactual)
- Itô diffusion and reverse-time SDE (Øksendal; Song et al.)
- Information geometry: Fisher-Rao metric and natural gradient (Amari)
- D4Explainer and GDSS reviewed as prior work

### §3 The Graph Explanation Manifold
- Bernoulli product parameterization of $\mathcal{M}_G$
- Diagonal Fisher-Rao metric derived
- **⚠ Key argument to defend:** Bernoulli independence assumption — either cite GDSS/D4Explainer as precedent, or flag as future extension

### §4 Geodesic Denoising via Natural Gradient (Algorithm 2)
- D4Explainer baseline vs. proposed replacement
- Natural gradient = Riemannian gradient on $(\mathcal{M}_G, g_\text{FR})$ — state as Lemma with Amari Ch.3 citation
- **⚠ Practical note:** Diagonal FIM makes this $O(|E|)$, not $O(|E|^2)$ — state explicitly

### §5 Attribution Decomposition (Algorithm 3)
- Geodesic path → martingale → per-edge $\phi_{ij}$
- Completeness property proven
- **⚠ Key argument to defend:** Martingale structure requires either formal proof or explicit discretization error bound

### §6 Experiments (D4Explainer Evaluation Framework)
- Datasets: BA-Shapes, BA-Community, Tree-Cycles, Tree-Grid
- Metrics: CF-ACC, Proximity, Validity, Sparsity
- Baselines: D4Explainer (primary), GNNExplainer, CF-GNNExplainer
- Ablation: natural gradient vs. standard gradient
- **⚠ Add:** One experiment measuring manifold-adherence of geodesic path vs. Euclidean path (discriminator score or FID along path)

### §7 Discussion
- Label preservation open problem
- Relaxing Bernoulli independence
- Scalability beyond diagonal FIM
- Connection to Shapley axioms

---

## 5. Pseudocode

### Algorithm 1 — D4Explainer Reverse Diffusion (Baseline)

```
INPUT:  Noisy graph A^(T), trained denoiser f_θ, GNN classifier f, steps T
OUTPUT: Counterfactual explanation e* = A^(0)_hat

FOR t = T DOWN TO 1:
    score_t = f_θ(A^(t), sigma=t)          # model forward pass [bsz, N, N, 1]
    A^(t-1) = denoise_step(A^(t), score_t) # Euclidean gradient step
    if f(A^(t-1)) != f(A^(T)):             # prediction flipped
        e* = threshold(A^(t-1))
        RETURN e*

RETURN threshold(A^(0)_hat)                # best available if no flip
```

### Algorithm 2 — Geodesic Denoising via Natural Gradient (Proposed)

```
INPUT:  A^(T), f_θ, f, T, step sizes {a_t}, epsilon=1e-6
OUTPUT: Geodesic path {θ^(t)}, explanation e*_geo

θ^(T) = A^(T)                              # initialize on manifold

FOR t = T DOWN TO 1:
    # 1. Forward pass — same as D4Explainer
    score_t = f_θ(θ^(t), sigma=t)          # [bsz, N, N, 1]

    # 2. Compute Euclidean gradient
    grad = ∇_θ L_cf(θ^(t), f)             # from loss_cf_exp

    # 3. Compute diagonal FIM and apply natural gradient
    theta = clamp(θ^(t), epsilon, 1-epsilon)
    inv_FIM = theta * (1 - theta)           # I(θ)^{-1} diagonal, O(|E|)
    nat_grad = grad * inv_FIM              # element-wise rescaling

    # 4. Update — geodesic step on M_G
    θ^(t-1) = θ^(t) - a_t * nat_grad

    STORE θ^(t-1) in path

e*_geo = threshold(θ^(0))
RETURN {θ^(t)}, e*_geo
```

### Algorithm 3 — Attribution Decomposition via Martingale Increments

```
INPUT:  Geodesic path {θ^(t)}_{t=0}^{T}, GNN classifier f, input graph G
OUTPUT: Per-edge attribution scores {φ_ij}, ranked edge list

# 1. Compute martingale values along path
FOR t = 0 TO T:
    M_t = f(θ^(t))                         # GNN prediction at step t

# 2. Compute per-edge attributions
FOR each edge (i,j) in E:
    φ_ij = 0
    FOR t = 0 TO T-1:
        ΔM_t = M_{t+1} - M_t
        total_move = Σ_{(a,b)} |θ^(t+1)[ab] - θ^(t)[ab]|
        w_ij^(t) = |θ^(t+1)[ij] - θ^(t)[ij]| / total_move
        φ_ij += ΔM_t * w_ij^(t)

# 3. Completeness check
Δ_total = f(e*_geo) - f(G)
ASSERT Σ_{(i,j)} φ_ij ≈ Δ_total          # holds by martingale property

# 4. Rank and return
ranked = SortDescending({φ_ij})
RETURN {φ_ij}, ranked
```

---

## 6. Implementation Plan

### Intervention Point in D4Explainer Training Loop

The natural gradient hooks into the training loop **between** `loss.backward()` and `optimizer.step()`. The loss functions themselves are unchanged.

```
loss = loss_dist + args.alpha_cf * loss_cf
loss.backward()                    # ← unchanged: computes ∇_θ L
# === YOUR ADDITIONS ===
register_natural_gradient_hook(score_batch)  # registered BEFORE backward
# ======================
optimizer.step()                   # ← now steps in natural gradient direction
```

### Function 1: Diagonal FIM Computation

```python
def compute_fim_diagonal(score, masks, epsilon=1e-6):
    """
    Compute diagonal Fisher Information Matrix for Bernoulli product manifold.

    For edge probability θ_ij, the Fisher information is:
        I(θ)_ij = 1 / (θ_ij * (1 - θ_ij))

    Args:
        score: list of [bsz, N, N, 1] tensors (model output edge probabilities)
        masks: list of [bsz, N, N] tensors (valid edge indicators)
        epsilon: clamp value to avoid singularity at boundary

    Returns:
        fim_list: list of [bsz, N, N] diagonal FIM tensors
    """
    fim_list = []
    for s, m in zip(score, masks):
        theta = s.squeeze(-1).clamp(epsilon, 1 - epsilon)
        fim = 1.0 / (theta * (1 - theta))   # I(θ)_ij diagonal
        fim = fim * m                         # zero out non-edges
        fim_list.append(fim)
    return fim_list
```

### Function 2: Natural Gradient Hook

```python
def register_natural_gradient_hook(score_batch, epsilon=1e-6):
    """
    Register a backward hook that rescales gradients by I(θ)^{-1}
    before backprop continues through the network.

    I(θ)^{-1}_ij = θ_ij * (1 - θ_ij)  [diagonal Bernoulli manifold]

    Call this immediately after model forward pass, inside the sigma loop.
    The hook fires automatically during loss.backward().

    Args:
        score_batch: [bsz, N, N, 1] model output tensor (requires_grad=True)
        epsilon: boundary clamp
    """
    theta = score_batch.squeeze(-1).clamp(epsilon, 1 - epsilon)
    inv_fim = theta * (1 - theta)              # I(θ)^{-1} diagonal

    def hook(grad):
        return grad * inv_fim.unsqueeze(-1)    # rescale incoming gradient

    score_batch.register_hook(hook)
```

### Where to Call in Training Loop

```python
# Inside: for i, sigma in enumerate(sigma_list):
score_batch = model(
    A=train_noise_adj_b_chunked[i].to(args.device),
    node_features=train_x_b_chunked[i].to(args.device),
    mask=mask.to(args.device),
    noiselevel=sigma,
)  # [bsz, N, N, 1]

# === ADD THIS LINE ===
register_natural_gradient_hook(score_batch)
# =====================

score.append(score_batch)
masks.append(mask)

# ... rest of loop unchanged ...

loss = loss_dist + args.alpha_cf * loss_cf
loss.backward()    # hook fires here, rescaling gradients in θ-space
optimizer.step()   # walks geodesic direction
```

### Function 3: Attribution Computation (Post-Training)

```python
def compute_geodesic_attribution(theta_path, gnn_model, graph, device):
    """
    Compute per-edge attribution scores from geodesic path.

    Args:
        theta_path: list of [N, N] tensors, the geodesic {θ^(t)}
        gnn_model:  frozen GNN classifier f
        graph:      original input graph (PyG Data object)
        device:     torch device

    Returns:
        phi: [N, N] tensor of attribution scores
        completeness_error: scalar, should be ~0
    """
    T = len(theta_path)
    M = []

    # Martingale values along geodesic
    for theta_t in theta_path:
        with torch.no_grad():
            pred = gnn_model(theta_t.to(device))
            M.append(pred)

    # Per-edge attribution
    N = theta_path[0].shape[0]
    phi = torch.zeros(N, N, device=device)

    for t in range(T - 1):
        delta_M = M[t + 1] - M[t]
        delta_theta = theta_path[t + 1] - theta_path[t]
        total_move = delta_theta.abs().sum() + 1e-10
        w = delta_theta.abs() / total_move
        phi += delta_M * w

    # Completeness check
    delta_total = M[-1] - M[0]
    completeness_error = (phi.sum() - delta_total).abs().item()

    return phi, completeness_error
```

---

## 7. Evaluation Protocol (Following D4Explainer)

| Metric | Definition | Target |
|---|---|---|
| **CF-ACC** | Fraction of explanations that flip GNN prediction | Higher is better |
| **Proximity** | Avg. edit distance between $G$ and $G'$ | Lower is better |
| **Validity** | In-distribution score (discriminator) | Higher is better |
| **Sparsity** | Fraction of edges retained in $G'$ | Lower is better |

**Datasets:** BA-Shapes, BA-Community, Tree-Cycles, Tree-Grid

**Primary baseline:** D4Explainer (Algorithm 1 above, same init and hyperparameters)

**Additional experiment (geometric validation):** Measure discriminator/FID score *along the denoising path* for D4Explainer vs. geodesic approach — this directly tests whether the geodesic path stays closer to the data manifold.

---

## 8. Open Questions and Known Vulnerabilities

1. **Bernoulli independence assumption** — edges in real graphs are not independent; the diagonal FIM is an approximation. Defended by precedent in GDSS and D4Explainer; flagged as future extension.
2. **Martingale structure** — strictly holds in continuous time; discretization introduces approximation error. Needs either a formal bound or explicit acknowledgment.
3. **Shapley analogy** — completeness property is in the *spirit* of Shapley efficiency, not a formal equivalence. Don't overclaim.
4. **Label preservation** — forward diffusion may not preserve $f(G)$; Fisher-Rao geometry could provide a principled label-preserving schedule. Strongest open research question.
