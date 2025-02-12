import torch
import numpy as np
# local imports
from .util import (
    Reduction,
)


def correct(output, target, topk=(1,)):
    """Computes how many correct outputs with respect to targets

    Does NOT compute accuracy but just a raw amount of correct
    outputs given target labels. This is done for each value in
    topk. A value is considered correct if target is in the topk
    highest values of output.
    The values returned are upperbounded by the given batch size

    [description]

    Arguments:
        output {torch.Tensor} -- Output prediction of the model
        target {torch.Tensor} -- Target labels from data

    Keyword Arguments:
        topk {iterable} -- [Iterable of values of k to consider as correct] (default: {(1,)})

    Returns:
        List(int) -- Number of correct values for each topk
    """
    maxk = max(topk)
    # Only need to do topk for highest k, reuse for the rest
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.item())
    return res


def accuracy_dl(model, dataloader, topk=(1,)):
    """Compute accuracy of a model over a dataloader for various topk

    Arguments:
        model {torch.nn.Module} -- Network to evaluate
        dataloader {torch.utils.data.DataLoader} -- Data to iterate over

    Keyword Arguments:
        topk {iterable} -- [Iterable of values of k to consider as correct] (default: {(1,)})

    Returns:
        List(float) -- List of accuracies for each topk
    """

    # Use same device as model
    device = next(model.parameters()).device

    accs = np.zeros(len(topk))

    for input, target in dataloader:
        input = input.to(device)
        target = target.to(device)
        output = model(input)

        accs += np.array(correct(output, target, topk))

    # Normalize over data length
    accs /= len(dataloader.dataset)

    return accs


def _accuracy(output, target, topk=(1,)):
    # from https://github.com/pytorch/examples/blob/0cb38ebb1b6e50426464b3485435c0c6affc2b65/imagenet/main.py#L436
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def accuracy(
    y_pred, 
    y_true,
    return_weights: bool = False,
    positive_class_weight: float = 1.0,
    batch_reduction: Reduction = "mean"
):
    maxk = max((1,))
    _, pred = y_pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred)).float().squeeze(0)

    if batch_reduction == "mean":
        return correct.mean()
    elif batch_reduction == "sum":
        return correct.sum()
    else:
        if return_weights:
            # Create a tensor of the same shape as y_true filled with zeros
            weights_per_sample = torch.zeros_like(y_true, dtype=torch.float)
            # If we want to balance the classes in binary classification then we can 
            # assign a weight to each sample based on the target value.
            weights_per_sample[y_true == 1] = positive_class_weight
            weights_per_sample[y_true == 0] = 1 / positive_class_weight
            # Return both the correct AND the weights
            return correct, weights_per_sample
        else:
            return correct
