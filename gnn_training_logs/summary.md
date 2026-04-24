# Base GNN Training Summary

CPU-only run on 2026-04-24. All checkpoints saved to `param/gnns/{dataset}_gcn.pt`; per-epoch CSV logs in `param/gnns/logs/`.

| Dataset        | Task | Epochs | Peak test acc | Final test acc | Per-epoch | Notes |
|----------------|------|--------|---------------|----------------|-----------|-------|
| Mutagenicity   | GC   | 300    | —             | 0.8160        | —         | Trained prior session |
| BA-3Motif      | GC   | 300    | 0.8125        | 0.7975        | ~1.0 s    | Stable plateau |
| NCI1           | GC   | 300    | 0.7760        | 0.7620        | ~3.0 s    | |
| BBBP           | GC   | 300    | 0.7000        | 0.6750        | ~1.5 s    | Severe overfit: train 93.7% / val 48.5% / test 67.5%. Faithful to repo defaults. |
| Cornell (web)  | NC   | 3000   | 0.6486        | 0.6216        | ~0.01 s   | Tiny test set (37 nodes); large val-loss drift, flat test acc |
| BA_shapes      | NC   | 10000  | 0.9857        | 0.9857        | ~0.02 s   | |
| Tree_Cycle     | NC   | 10000  | 0.9659        | 0.9659        | ~0.015 s  | |
| Tree_Grids     | NC   | 10000  | 0.9194        | 0.9113        | ~0.025 s  | |

GC = graph classification, NC = node classification. Peak = best value among the periodic `Test set results` evaluations during training; Final = last epoch's evaluation, which is what `torch.save` persists (scripts do not track a best-val checkpoint).

## Deviations / fixes applied this session

- **`gnns/bbbp_gnn.py` save-path bug**: default `--model_path` was `{gnns/}param/gnns` (missing `..`), so the checkpoint saved to `gnns/param/gnns/bbbp_gcn.pt` instead of the repo's `param/gnns/`. Patched the default to match the other GC scripts; the already-trained `bbbp_gcn.pt` was relocated to the correct location.
- Environment: used conda env `d4explainer` (PyTorch 2.2.2, PyG 2.6.1), not `d4explainer-geo` as historically noted.

## Caveats for appendix reporting

- These are repo-default hyperparameters, not paper-reported optimal configs. Serves as a baseline only. BBBP's overfitting (no regularisation tuning) and Cornell's val/test divergence reflect the defaults, not a training failure.
- Final-epoch checkpoints are used downstream rather than best-val checkpoints because that is what the upstream training scripts save.
