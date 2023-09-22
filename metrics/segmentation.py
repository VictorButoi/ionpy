# torch imports
import torch
from torch import Tensor

# misc imports
from pydantic import validate_arguments
from typing import Optional, Union, List


from .util import (
    _metric_reduction,
    _inputs_as_onehot,
    _inputs_as_longlabels,
    InputMode,
    Reduction,
)


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode, 
        from_logits=from_logits, 
        discretize=True
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    pred_amounts = (y_pred == 1.0).sum(dim=-1)
    true_amounts = (y_true == 1.0).sum(dim=-1)
    cardinalities = pred_amounts + true_amounts

    dice_scores = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float().cpu()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    score = _metric_reduction(
        dice_scores,
        reduction=reduction,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )

    return score


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_longlabels(
        y_pred, y_true, mode, from_logits=from_logits, discretize=True
    )
    correct = y_pred == y_true
    return correct.float().mean()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pixel_precision(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_longlabels(
        y_pred, y_true, mode, from_logits=from_logits, discretize=True
    )

    # Get tensor of ones like y_pred
    one_y_pred = torch.ones_like(y_pred)

    correct = (one_y_pred == y_true)
    return correct.float().mean()


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def balanced_pixel_accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    from_logits: bool = False,
    ignore_labels: List[int] = []
) -> Tensor:
    y_pred, y_true = _inputs_as_longlabels(
        y_pred, y_true, mode, from_logits=from_logits, discretize=True
    )

    # Get unique labels in y_true
    unique_labels = torch.unique(y_true).tolist()

    # Remove labels to be ignored
    unique_labels = [label for label in unique_labels if label not in ignore_labels]
    accuracies = []
    
    for label in unique_labels:
        # Create a mask for the current label
        label_mask = (y_true == label).bool()
        
        # Extract predictions and ground truth for pixels belonging to the current label
        label_pred = y_pred[label_mask]
        label_true = y_true[label_mask]
        
        # Calculate accuracy for the current label
        correct_label = (label_pred == label_true).float()
        accuracies.append(correct_label.mean())
        
    # Calculate balanced accuracy by averaging individual label accuracies
    balanced_accuracy = torch.tensor(accuracies).mean()
    
    return balanced_accuracy


def pixel_mse(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    # per_channel: bool = False,
    from_logits: bool = False,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_index: Optional[int] = None,
    weights: Optional[Union[Tensor, List]] = None,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    # if per_channel:
    # NOTE: Each channel is weighted equally because of the mean reduction
    correct = (y_pred - y_true).square().mean(dim=-1)

    return _metric_reduction(
        correct,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_dice_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    ignore_empty_labels: bool = False,
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    
    y_pred, y_true = _inputs_as_onehot(
        y_pred, 
        y_true, 
        mode=mode,
        from_logits=from_logits
    )
    assert y_pred.shape == y_true.shape

    intersection = torch.sum(y_pred * y_true, dim=-1)

    if square_denom:
        pred_amounts = y_pred.square().sum(dim=-1)
        true_amounts = y_true.square().sum(dim=-1)
    else:
        pred_amounts = y_pred.sum(dim=-1)
        true_amounts = y_true.sum(dim=-1)
    
    cardinalities = pred_amounts + true_amounts
    soft_dice_score = (2 * intersection + smooth) / (cardinalities + smooth).clamp_min(eps)

    if ignore_empty_labels:
        existing_label = (true_amounts > 0).float().cpu()
        if weights is None:
            weights = existing_label
        else:
            weights = weights * existing_label

    score = _metric_reduction(
        soft_dice_score,
        reduction=reduction,
        weights=weights,
        ignore_empty_labels=ignore_empty_labels,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )
    
    if score > 0.5:
        print("dice score: ", score)
        print("ypred: ", y_pred.shape)
        print("ytrue: ", y_true.shape)
        print("intersection: ", intersection)
        print("cardinalties: ", cardinalities)
        print("cardinalties for ypred: ", y_pred.sum(dim=-1))
        print("cardinalties for ytrue: ", y_true.sum(dim=-1))

    return score
    

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def soft_jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    square_denom: bool = True,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:
    y_pred, y_true = _inputs_as_onehot(
        y_pred, y_true, mode=mode, from_logits=from_logits
    )
    assert y_pred.shape == y_true.shape

    intersection = torch.sum(y_pred * y_true, dim=-1)

    if square_denom:
        cardinalities = y_pred.square().sum(dim=-1) + y_true.square().sum(dim=-1)
    else:
        cardinalities = y_pred.sum(dim=-1) + y_true.sum(dim=-1)

    union = cardinalities - intersection

    score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )


def jaccard_score(
    y_pred: Tensor,
    y_true: Tensor,
    mode: InputMode = "auto",
    smooth: float = 1e-7,
    eps: float = 1e-7,
    reduction: Reduction = "mean",
    batch_reduction: Reduction = "mean",
    weights: Optional[Union[Tensor, List]] = None,
    ignore_index: Optional[int] = None,
    from_logits: bool = False,
) -> Tensor:

    y_pred, y_true = _inputs_as_onehot(
        y_pred,
        y_true,
        mode=mode,
        from_logits=from_logits,
        discretize=True,
    )

    intersection = torch.logical_and(y_pred == 1.0, y_true == 1.0).sum(dim=-1)
    cardinalities = (y_pred == 1.0).sum(dim=-1) + (y_true == 1.0).sum(dim=-1)
    union = cardinalities - intersection

    score = (intersection + smooth) / (union + smooth).clamp_min(eps)

    return _metric_reduction(
        score,
        reduction=reduction,
        weights=weights,
        ignore_index=ignore_index,
        batch_reduction=batch_reduction,
    )
