# torch imports
import torch
from torch import Tensor
import torch.nn.functional as F

# misc imports
import einops as E
from pydantic import validate_arguments
from typing import Literal, Optional, Tuple, Union

InputMode = Literal["binary", "multiclass", "onehot", "auto"]
Reduction = Union[None, Literal["mean", "sum"]]


def expand_to_4d(y_pred):
    num_dims = len(y_pred.shape)
    
    if num_dims == 1:
        return y_pred.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 1, W)
    if num_dims == 2:  # Shape: (H, W)
        return y_pred.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    elif num_dims == 3:  # Shape: (C, H, W)
        return y_pred.unsqueeze(0)  # Shape: (1, C, H, W)
    elif num_dims == 4:  # Shape: (B, C, H, W)
        return y_pred  # No expansion needed
    else:
        raise ValueError(f"Input y_pred must have 1, 2, 3, or 4 dimensions, but got {num_dims}.")


def hard_max(x: Tensor):
    """
    argmax + onehot
    """
    N = len(x.shape)
    order = (0, N - 1, *[i for i in range(1, N - 1)])
    return F.one_hot(torch.argmax(x, dim=1), num_classes=x.shape[1]).permute(order)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _infer_mode(y_pred: Tensor, y_true: Tensor,) -> InputMode:
    batch_size, num_classes = y_pred.shape[:2]

    if y_pred.shape == y_true.shape:
        return "binary" if num_classes == 1 else "onehot"
    else:
        return "multiclass"


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _inputs_as_onehot(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    discretize: bool = False,
) -> Tuple[Tensor, Tensor]:
    
    # Expends these potentially to account for missing dimensions.
    y_pred = expand_to_4d(y_pred)
    y_true = expand_to_4d(y_true)
    batch_size, num_classes = y_pred.shape[:2]

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    if from_logits:
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1

        if mode == "binary":
            # y_pred = F.logsigmoid(y_pred.float()).exp()
            y_pred = torch.sigmoid(y_pred.float())
        elif mode in ("multiclass", "onehot"):
            # y_pred = F.log_softmax(y_pred.float(), dim=1).exp()
            y_pred = torch.softmax(y_pred.float(), dim=1)

    if discretize:
        if mode == "binary":
            y_pred = torch.round(y_pred).clamp_min(0.0).clamp_max(1.0)
            y_true = torch.round(y_true).clamp_min(0.0).clamp_max(1.0)
        elif mode == "onehot":
            y_pred = hard_max(y_pred)
            y_true = hard_max(y_true)
        elif mode == "multiclass":
            y_pred = hard_max(y_pred)

    if mode == "binary":
        y_true = y_true.reshape(batch_size, 1, -1)
        y_pred = y_pred.reshape(batch_size, 1, -1)

    elif mode == "onehot":
        y_true = y_true.reshape(batch_size, num_classes, -1)
        y_pred = y_pred.reshape(batch_size, num_classes, -1)

    elif mode == "multiclass":
        y_pred = y_pred.reshape(batch_size, num_classes, -1)
        y_true = y_true.reshape(batch_size, -1)
        if y_true.dtype != torch.long:
            y_true_long = y_true.long()
        else:
            y_true_long = y_true
        y_true = F.one_hot(y_true_long, num_classes).permute(0, 2, 1)

    assert y_pred.shape == y_true.shape
    return y_pred.float(), y_true.float()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _inputs_as_longlabels(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    discretize: bool = False,
) -> Tuple[Tensor, Tensor]:

    # Expends these potentially to account for missing dimensions.
    y_pred = expand_to_4d(y_pred)
    y_true = expand_to_4d(y_true)
    batch_size, num_classes = y_pred.shape[:2]

    if mode == "auto":
        if y_pred.shape == y_true.shape:
            mode = "binary" if num_classes == 1 else "onehot"
        else:
            mode = "multiclass"

    if discretize:
        if mode == "binary":
            y_pred = torch.round(y_pred).clamp_min(0.0).clamp_max(1.0)
        else:
            y_pred = hard_max(y_pred)

    if mode == "binary":
        if from_logits:
            y_pred = F.logsigmoid(y_pred.float()).exp()
        y_pred = torch.round(y_pred).clamp_max(1).clamp_min(0).long()
    else:
        if from_logits:
            y_pred = F.log_softmax(y_pred.float(), dim=1).exp()
        batch_size, n_classes = y_pred.shape[:2]
        y_pred = y_pred.view(batch_size, n_classes, -1)
        y_pred = torch.argmax(y_pred, dim=1)

        if mode == "onehot":
            y_true = torch.argmax(y_true, dim=1)
        y_true = y_true.view(batch_size, -1)

    assert y_pred.shape == y_true.shape
    return y_pred, y_true.long()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def _metric_reduction(
    loss: Tensor,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    weights: Optional[Union[Tensor, list]] = None,
    ignore_index: Optional[int] = None,
) -> Tensor:

    if len(loss.shape) != 2:
        raise ValueError(
            f"Reduceable Tensor must have two dimensions, batch & channels, got {loss.shape} instead"
        )

    batch, channels = loss.shape

    if ignore_index is not None:
        assert 0 <= ignore_index < channels, "ignore_index must be in [0, channels)"
        uni_weights = torch.Tensor([1.0 if i != ignore_index else 0.0 for i in range(channels)])
        if weights is None:
            weights = uni_weights
        else:
            weights = weights * uni_weights

    if weights is not None:
        if isinstance(weights, list):
            weights = torch.Tensor(weights)
        # Identical weights for each item in the batch.
        if len(weights.shape) == 1:
            assert (
                len(weights) == channels
            ), f"Weights must match number of channels {len(weights)} != {channels}"
            weights = E.repeat(weights, "C -> B C", C=channels, B=batch)
        # Per batch item weights.
        elif len(weights.shape) == 2:
            assert (
                weights.shape[1] == channels
            ), f"Weights must match number of channels {weights.shape[1]} != {channels}"
        else:
            raise NotImplementedError("Haven't implemented weighting scheme when 3 dims.")
    else:
        weights = torch.ones(batch, channels)

    # Apply weights.
    loss *= weights.type(loss.dtype).to(loss.device)

    # Determine the number of classes to reduce over.
    if ignore_empty_labels:
        N = (weights.sum(dim=1) > 0).float().to(loss.device)
    else:
        N = channels
        if ignore_index is not None:
            N -= 1

    # Reduce over the classes.
    if reduction == "mean":
        loss = (1 / N) * loss.sum(dim=-1)
    elif reduction == "sum":
        loss = loss.sum(dim=-1)
    else:
        raise ValueError(f"Unknown reduction {reduction}")

    # Reduce over the examples in the batch.
    if batch_reduction == "sum":
        return loss.sum(dim=0)
    elif batch_reduction == "mean":
        return loss.mean(dim=0)
    else:
        return loss


def batch_channel_flatten(x: Tensor) -> Tensor:
    batch_size, n_channels, *_ = x.shape
    return x.view(batch_size, n_channels, -1)