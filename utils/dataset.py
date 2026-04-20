import os

import numpy as np
from torch.utils.data import Sampler

from datasets import NCI1, BA3Motif, Mutagenicity, SynGraphDataset, WebDataset, bbbp


class SizeBucketedBatchSampler(Sampler):
    """Batches graphs by node count so each batch pads to a tight N.

    Graphs are sorted by num_nodes then sliced into batches of size ``batch_size``.
    Per epoch, the *order* of batches is shuffled but membership is fixed — this
    keeps padded N close to the max node count of each bucket and prevents a
    single large outlier graph from inflating every batch it lands in.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        sizes = np.array([int(dataset[i].num_nodes) for i in range(len(dataset))])
        self.sorted_indices = np.argsort(sizes, kind="stable")

    def __iter__(self):
        batches = [
            self.sorted_indices[i : i + self.batch_size].tolist()
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


def filter_by_max_size(dataset, max_nodes):
    """Return indices of graphs with num_nodes <= max_nodes (caller slices the dataset)."""
    return [i for i in range(len(dataset)) if int(dataset[i].num_nodes) <= max_nodes]


def get_datasets(name, root="data/"):
    """
    Get preloaded datasets by name
    :param name: name of the dataset
    :param root: root path of the dataset
    :return: train_dataset, test_dataset, val_dataset
    """
    if name == "mutag":
        folder = os.path.join(root, "MUTAG")
        train_dataset = Mutagenicity(folder, mode="training")
        test_dataset = Mutagenicity(folder, mode="testing")
        val_dataset = Mutagenicity(folder, mode="evaluation")
    elif name == "NCI1":
        folder = os.path.join(root, "NCI1")
        train_dataset = NCI1(folder, mode="training")
        test_dataset = NCI1(folder, mode="testing")
        val_dataset = NCI1(folder, mode="evaluation")
    elif name == "ba3":
        folder = os.path.join(root, "BA3")
        train_dataset = BA3Motif(folder, mode="training")
        test_dataset = BA3Motif(folder, mode="testing")
        val_dataset = BA3Motif(folder, mode="evaluation")
    elif name == "BA_shapes":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="BA_shapes")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="BA_shapes")
        train_dataset = SynGraphDataset(folder, mode="training", name="BA_shapes")
    elif name == "Tree_Cycle":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="Tree_Cycle")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="Tree_Cycle")
        train_dataset = SynGraphDataset(folder, mode="training", name="Tree_Cycle")
    elif name == "Tree_Grids":
        folder = os.path.join(root)
        test_dataset = SynGraphDataset(folder, mode="testing", name="Tree_Grids")
        val_dataset = SynGraphDataset(folder, mode="evaluating", name="Tree_Grids")
        train_dataset = SynGraphDataset(folder, mode="training", name="Tree_Grids")
    elif name == "bbbp":
        folder = os.path.join(root, "bbbp")
        dataset = bbbp(folder)
        test_dataset = dataset[:200]
        val_dataset = dataset[200:400]
        train_dataset = dataset[400:]
    elif name == "cornell":
        folder = os.path.join(root)
        test_dataset = WebDataset(folder, mode="testing", name=name)
        val_dataset = WebDataset(folder, mode="evaluating", name=name)
        train_dataset = WebDataset(folder, mode="training", name=name)
    else:
        raise ValueError
    return train_dataset, val_dataset, test_dataset
