import torch
from torch import nn


class BestValTracker:
    """Tracks best-val checkpoint by validation accuracy and triggers early stopping.

    Keeps a CPU-side copy of `model.state_dict()` from the epoch with the
    highest validation accuracy seen so far, plus the test accuracy
    reported at that epoch. Counts consecutive non-improving epochs against
    `patience`. Val accuracy is preferred over val loss because cross-
    entropy minima can occur far from accuracy maxima on small/noisy val
    sets, and accuracy is a more direct proxy for what we report on test.
    """

    def __init__(self, patience):
        self.patience = patience
        self.best_val_acc = -float("inf")
        self.best_state = None
        self.best_test_acc = None
        self.best_epoch = -1
        self.no_improve = 0

    def update(self, epoch, val_acc, test_acc, model):
        val_acc = float(val_acc)
        improved = val_acc > self.best_val_acc
        if improved:
            self.best_val_acc = val_acc
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_test_acc = float(test_acc) if test_acc is not None else None
            self.best_epoch = epoch
            self.no_improve = 0
        else:
            self.no_improve += 1
        return improved

    @property
    def should_stop(self):
        return self.no_improve >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def make_val_mask_from_train(data, val_frac=0.1, seed=0):
    """If `data` has no usable val_mask, carve `val_frac` of train_mask out as val.

    The carved indices are flipped off in train_mask so train and val are disjoint.
    test_mask is never touched. Deterministic given `seed`.
    """
    has_val = hasattr(data, "val_mask") and data.val_mask is not None and bool(data.val_mask.any())
    if has_val:
        return
    train_idx = data.train_mask.nonzero(as_tuple=True)[0]
    if len(train_idx) == 0:
        raise ValueError("train_mask is empty; cannot carve val from it.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_idx), generator=g)
    n_val = max(1, int(len(train_idx) * val_frac))
    val_idx = train_idx[perm[:n_val]]
    val_mask = torch.zeros_like(data.train_mask)
    val_mask[val_idx] = True
    data.val_mask = val_mask
    data.train_mask = data.train_mask.clone()
    data.train_mask[val_idx] = False


def Gtrain(train_loader, model, optimizer, device, criterion=nn.MSELoss()):
    """
    General training function for graph classification
    :param train_loader: DataLoader
    :param model: model
    :param optimizer: optimizer
    :param device: device
    :param criterion: loss function (default: MSELoss)
    """
    model.train()
    loss_all = 0
    criterion = criterion

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def Gtest(test_loader, model, device, criterion=nn.L1Loss(reduction="mean")):
    """
    General test function for graph classification
    :param test_loader: DataLoader
    :param model: model
    :param device: device
    :param criterion: loss function (default: L1Loss)
    :return: error, accuracy
    """
    model.eval()
    error = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(
                data.x,
                data.edge_index,
                data.batch,
            )

            error += criterion(output, data.y) * data.num_graphs
            correct += float(output.argmax(dim=1).eq(data.y).sum().item())

        return error / len(test_loader.dataset), correct / len(test_loader.dataset)
