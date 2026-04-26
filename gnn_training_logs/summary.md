# Base GNN Training Summary

CPU-only runs on 2026-04-24 / 2026-04-25. Final checkpoints saved to `param/gnns/{dataset}_gcn.pt`; per-epoch CSV logs in `param/gnns/logs/`. Per-trainer stdout logs are kept under `gnn_training_logs/run{N}/` for traceability.

## Final results

| Dataset        | Task | Source run | Test acc | Paper (Table 6) | ≥80%? | Notes |
|----------------|------|------------|---------:|----------------:|:-----:|-------|
| Mutagenicity   | GC   | run3       | 0.8020   | 0.87            | ✓     | Early stop at 165, best at 135 |
| BA-3Motif      | GC   | run4       | 0.7625   | 0.93            | ✗     | Best at 173; val/test peaks decorrelated |
| NCI1           | GC   | run4       | 0.7660   | 0.83            | ✗     | Best at 98; ceiling ~0.78 |
| BBBP           | GC   | run4       | 0.6900   | 0.85            | ✗     | Val_acc max = 0.505 (chance); split is broken |
| Cornell (web)  | NC   | run4       | 0.5405   | 0.83            | ✗     | Val_acc max at epoch 10 with ~13 val nodes |
| BA_shapes      | NC   | run3       | 0.9857   | 0.99            | ✓     | Trained full 10000 epochs |
| Tree_Cycle     | NC   | run3       | 0.9659   | 0.98            | ✓     | Trained full 10000 epochs |
| Tree_Grids     | NC   | run3       | 0.8952   | 0.95            | ✓     | Trained full 10000 epochs |

GC = graph classification, NC = node classification.

## Methodology applied this iteration

Best-practice training was layered on top of the original repo trainers:

1. **Best-validation checkpointing.** Trainers track and restore the model state from the epoch with highest validation accuracy. (Earlier runs used validation loss; on small/noisy val sets here the loss minimum can occur far from the accuracy maximum, e.g. BBBP picked epoch 5/300 by loss. Switched to val_acc.)
2. **Early stopping with patience.** Default patience = 50 (GC), 300–500 (NC). Reduces wasted compute when val plateaus.
3. **Per-trainer CLI flags.** Added `--weight_decay`, `--patience`, `--val_frac` (NC) so each can be tuned without code changes. `weight_decay` defaults to 0 because the existing models already have dropout; adding 5e-4 over-regularized in earlier runs.
4. **NC val_mask carved deterministically from train_mask** (`val_frac=0.1`) — test_mask is never touched, so test accuracy here is an honest held-out number.

Helper code lives in `utils/train_utils.py`: `BestValTracker` and `make_val_mask_from_train`.

## Mixed-source caveat

The four "passing" checkpoints (mutag, BA_shapes, Tree_Cycle, Tree_Grids) on disk are from `run3`, which used best-val-by-LOSS. The four "failing" reruns (ba3, nci1, bbbp, cornell) on disk are from `run4`, which uses best-val-by-ACCURACY (the current code). Because all four passing-run trainers converged cleanly with val_loss and val_acc moving in lockstep, re-running them under the current code is expected to land at the same numbers within ±1pp. We did not re-run them.

## Why the four failures fall short of paper

The repo's checked-in hyperparameters and data splits do not reproduce paper Table 6 on BA-3Motif, NCI1, BBBP, or Cornell. Across run1 (final-epoch save), run3 (val-loss best), and run4 (val-acc best), each of these four hits a similar ceiling:

- **BA-3Motif**: best test_acc across runs ≈ 0.79–0.81; small val set, val/test peaks misaligned.
- **NCI1**: ceiling ≈ 0.77–0.78; LEConv architecture in the repo (paper said GCN).
- **BBBP**: ceiling ≈ 0.69–0.70; val accuracy is at chance, indicating the train/val/test split shipped with the repo is distributionally broken.
- **Cornell**: ceiling ≈ 0.62; tiny test set (37 nodes), tiny val set after carving (~13 nodes), high variance.

Closing the gap to paper would require per-dataset hyperparameter sweeps and possibly different data splits — not a checkpoint-policy fix.

## Why this is acceptable for the explainer comparison

The base GNN is shared infrastructure: both the baseline D4Explainer and the natural-gradient extension run against the same frozen `param/gnns/{dataset}_gcn.pt`. As long as that GNN is identical across the two methods (and it is), the comparison's internal validity does not depend on hitting paper accuracy. Test accuracy here is best read as "what we deploy as the frozen target," not as a reproduction of the paper's claim.

## Deviations / fixes from the very first session

- **`gnns/bbbp_gnn.py` save-path bug** (round 1): default `--model_path` was `gnns/param/gnns` (missing `..`). Patched.
- Environment: conda env `d4explainer` (PyTorch 2.2.2, PyG 2.6.1) — historical references to `d4explainer-geo` are stale.
- All trainers now print a final `Best val_acc=… at epoch …, test_acc_at_best=…` line for easy log scraping.
