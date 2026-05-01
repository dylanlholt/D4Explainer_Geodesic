import argparse
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import torch
from torch_geometric.loader import DataLoader
from constants import feature_dict, task_type, dataset_choices
from explainers import *
from gnns import *
from utils.dataset import filter_by_max_size, get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--root", type=str, default="results/", help="Result directory.")
    parser.add_argument("--dataset", type=str, default="Tree_Cycle", choices=dataset_choices)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--task", type=str, default="nc")

    parser.add_argument("--train_batchsize", type=int, default=32)
    parser.add_argument("--test_batchsize", type=int, default=32)
    parser.add_argument("--sigma_length", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=800)
    parser.add_argument("--feature_in", type=int)
    parser.add_argument("--data_size", type=int, default=-1)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha_cf", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prob_low", type=float, default=0.0)
    parser.add_argument("--prob_high", type=float, default=0.4)
    parser.add_argument("--sparsity_level", type=float, default=2.5)

    parser.add_argument("--normalization", type=str, default="instance")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layers_per_conv", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--cat_output", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False)
    parser.add_argument("--noise_mlp", type=bool, default=True)
    parser.add_argument("--simplified", type=bool, default=False)

    parser.add_argument("--memory_efficient", action="store_true", default=False, help="Opt into per-sigma BCE backward + single-sigma CF approximation. Off = paper-faithful batched-sigma loop (single backward, exact CF averaged over all sigmas). Turn on only when the paper loop OOMs.")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Enable mixed precision (fp16) training. Only applied in --memory_efficient mode.")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing in Powerful to trade compute for memory.")
    parser.add_argument("--debug_shapes", action="store_true", default=False, help="Print padded N, real node counts, and CUDA memory for first 5 batches of each epoch.")
    parser.add_argument("--size_bucketed", action="store_true", default=False, help="Use a node-count-bucketed batch sampler so each batch pads to a tight N (reduces peak memory on size-heterogeneous datasets like Mutagenicity).")
    parser.add_argument("--max_graph_size", type=int, default=None, help="Drop graphs with num_nodes > max_graph_size from ALL splits (train/val/test) before training. Hardware-driven data cap applied uniformly so train and eval distributions match. Documented as a data deviation; loss/optimizer unchanged.")
    parser.add_argument("--natural_gradient", action="store_true", default=False, help="[Extension] Apply diagonal Fisher-Rao natural-gradient rescaling to score tensors. Off = paper baseline.")
    parser.add_argument("--nat_grad_eps", type=float, default=1e-6, help="[Extension] Boundary clamp for θ in natural-gradient hook to prevent vanishing/exploding gradient at θ∈{0,1}.")
    parser.add_argument("--nat_grad_cap", type=float, default=100.0, help="[Extension] Upper bound on the 1/(θ(1-θ)) rescaling factor. Caps gradient amplification at extreme logits to bound update size.")
    parser.add_argument("--nat_grad_cf_only", action="store_true", default=False, help="[Extension] Restrict natural-gradient hook to the CF loss pathway only; leave BCE/denoising gradient unmodified. Per CONTEXT.md §2.1 the Fisher-Rao framing is about the explanation manifold (CF domain), not the reconstruction objective. No effect unless --natural_gradient is also set.")
    parser.add_argument("--run_name", type=str, default=None, help="Optional subfolder under results/{dataset}/ for this run's artifacts (best_model.pth, config.json, metrics.jsonl). Use to keep baseline and extension runs separate.")

    return parser.parse_args()


args = parse_args()
args.noise_list = None

args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
args.feature_in = feature_dict[args.dataset]
args.task = task_type[args.dataset]
train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)

train_dataset = train_dataset[: args.data_size]
if args.max_graph_size is not None:
    for split_name, split in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        keep = filter_by_max_size(split, args.max_graph_size)
        kept, total = len(keep), len(split)
        print(f"[max_graph_size={args.max_graph_size}] {split_name}: keeping {kept}/{total} graphs")
        if split_name == "train":
            train_dataset = train_dataset[keep]
        elif split_name == "val":
            val_dataset = val_dataset[keep]
        else:
            test_dataset = test_dataset[keep]
gnn_path = f"param/gnns/{args.dataset}_{args.gnn_type}.pt"
explainer = DiffExplainer(args.device, gnn_path)

# Train D4Explainer over train_dataset and evaluate
explainer.explain_graph_task(args, train_dataset, val_dataset)

# Test D4Explainer on test_dataset
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
for graph in test_loader:
    explanation, y_ori, y_exp, modif_r = explainer.explain_evaluation(args, graph)