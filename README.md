# D4Explainer + Fisher-Rao Natural Gradient Extension

This fork extends **D4Explainer** (Chen et al., NeurIPS 2023) with a Fisher-Rao natural-gradient update rule for the diffusion-based explainer. The original README is preserved below for side-by-side comparison.

---

## Extension: Fisher-Rao Natural Gradient

### Method
- Each candidate explanation is parameterized as edge probabilities $\theta \in [0,1]^{|E|}$, which defines a product-Bernoulli distribution over graphs — a statistical manifold $\mathcal{M}_G$.
- The natural metric on $\mathcal{M}_G$ is Fisher-Rao, not Euclidean. Its Fisher information matrix is diagonal: $\mathcal{I}(\theta)_{ij} = 1 / (\theta_{ij}(1-\theta_{ij}))$.
- We replace the Euclidean gradient step with a natural-gradient step, so each denoising update follows the geodesic on $\mathcal{M}_G$.
- Implementation: a backward hook registered on the score tensor rescales $\partial\mathcal{L}/\partial\theta$ in-place. Loss functions, model architecture, and training data are all unchanged.
- The extension is gated behind `--natural_gradient`; with the flag off, training is byte-identical to the baseline.

### Loss (unchanged) and Natural-Gradient Update

The D4Explainer loss is kept as-is:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{dist}}(\theta) + \alpha_{\text{cf}} \cdot \mathcal{L}_{\text{cf}}(\theta)$$

The **update rule** is what changes. Standard (Euclidean) gradient descent:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \, \nabla_\theta \mathcal{L}$$

Natural gradient descent on the product-Bernoulli manifold uses the inverse Fisher information. Because the metric is diagonal, this simplifies to an element-wise rescaling:

$$\tilde{\nabla}_\theta \mathcal{L}_{ij} = \theta_{ij}(1 - \theta_{ij}) \cdot \nabla_\theta \mathcal{L}_{ij}$$

$$\theta^{(t+1)} = \theta^{(t)} - \eta \cdot \tilde{\nabla}_\theta \mathcal{L}$$

To avoid vanishing updates at $\theta \in \\{0, 1\\}$, the rescaling uses $\theta \leftarrow \mathrm{clip}(\theta, \epsilon, 1-\epsilon)$ with $\epsilon$ controlled by `--nat_grad_eps` (default `1e-6`).

---

## Reproducing This Project

All extension commands mirror the baseline D4Explainer command exactly — only `--natural_gradient` and `--run_name` differ, so results are directly comparable.

### Smoke tests (verify pipeline, ~1 minute)

Run a single epoch on a small dataset to confirm the environment, data loaders, and hook registration all work:

```bash
# Baseline smoke test
python main.py --dataset BA_shapes --epoch 1 --run_name smoke_baseline

# Extension smoke test
python main.py --dataset BA_shapes --epoch 1 --natural_gradient --run_name smoke_natgrad
```

Both should complete without errors and write `config.json`, `metrics.jsonl`, and `best_model.pth` under `results/BA_shapes/{run_name}/`.

### Paired baseline vs. extension runs

For every experiment, run the baseline and the extension with **matched hyperparameters** — only the `--natural_gradient` flag and `--run_name` should differ:

```bash
# Baseline (original D4Explainer)
python main.py --dataset mutag --run_name baseline

# Extension (Fisher-Rao natural gradient)
python main.py --dataset mutag --natural_gradient --run_name natgrad
```

Each run writes:
- `results/{dataset}/{run_name}/config.json` — all CLI args
- `results/{dataset}/{run_name}/metrics.jsonl` — one JSON record per epoch
- `results/{dataset}/{run_name}/best_model.pth` — best checkpoint

Compare the two `metrics.jsonl` files to isolate the contribution of the natural-gradient update.

### Tested environment

The extension itself adds no new Python dependencies — `explainers/natural_gradient.py` is a backward hook using only existing torch ops. Tested against **PyTorch 2.x + torch-geometric 2.x** (conda env `d4explainer-geo`). The pinned `torch==1.10.1` in `requirements.txt` is the original paper's environment and predates this fork.

### Extension CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--natural_gradient` | off | Enable Fisher-Rao natural-gradient rescaling on score tensors. |
| `--nat_grad_eps` | `1e-6` | Boundary clamp for $\theta$ to prevent vanishing updates at $\theta \in \\{0, 1\\}$. |
| `--run_name` | `None` | Subfolder under `results/{dataset}/` so baseline and extension runs do not clobber each other. |

---

## Paper Hyperparameters (D4Explainer Table 7, Appendix E.3)

The repo's CLI defaults diverge from the values reported in the paper. Use the following per-dataset settings to reproduce paper-comparable numbers:

| Dataset | `--n_hidden` | `--num_layers` | `--train_batchsize` | `--alpha_cf` |
|---|---|---|---|---|
| BA-shapes | 64 | 6 | 4 | 0.005 |
| Tree-Cycle | 64 | 6 | 32 | 0.1 |
| Tree-Grids | 128 | 8 | 32 | 0.05 |
| Cornell | 128 | 6 | 4 | 0.05 |
| BA-3Motif | 128 | 6 | 32 | 0.05 |
| Mutag | 64 | 6 | 2 | 0.001 |
| BBBP | 128 | 6 | 16 | 0.005 |
| NCI1 | 128 | 6 | 32 | 0.01 |

The paper reports training for `--epoch 1500` across all datasets.

> **Note:** The repo's `--dataset mutag` flag loads **Mutagenicity** (4,337 graphs, max ~417 nodes), not the classic TU MUTAG (188 graphs, max 28). The D4Explainer authors kept the imprecise label; the paper's reported "MUTAG" numbers are trained on Mutagenicity.

Example paper-faithful command:
```bash
python main.py --dataset mutag --train_batchsize 2 --alpha_cf 0.001 --epoch 1500 --run_name baseline_paper
```

---

# Original Paper Implementation

# D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion [NeurIPS 2023]
This is the Pytorch implementation of " D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion"
## Requirements

- `torch==1.10.1`
- `torch-geometric==2.0.4`
- `numpy==1.24.2`
- `pandas==1.5.3`
- `networkx==3.0`

Refer to `requirements.txt` for more details.


## Dataset

Download the datasets from [here](https://drive.google.com/drive/folders/1pwmeST3zBcSC34KbAL_Wvi-cFtufAOCE?usp=sharing) to `data/`

**Datasets Included:**

- Node classification: `BA_shapes`; `Tree_Cycle`; `Tree_Grids`; `cornell`
- Graph classification: `mutag`; `ba3`; `bbbp`; `NCI1`

## Train Base GNNs
```
cd gnns
python ba3motif_gnn.py
python bbbp_gnn.py
python mutag_gnn.py
python nci1_gnn.py
python synthetic_gnn.py --data_name Tree_Cycle
python synthetic_gnn.py --data_name BA_shapes
python tree_grids_gnn.py
python web_gnn.py
```


## Train and Evaluate D4Explainer
For example, to train D4Explainer on Mutag, run:
```
python main.py --dataset mutag
```


## Evaluation of Other Properties

- In-distribution: `python -m evaluation.ood_evaluation`
- Robustness: `python -m evaluation.robustness`
