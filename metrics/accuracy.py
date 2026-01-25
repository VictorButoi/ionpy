import torch
import numpy as np
from torch import Tensor
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
    from_logits: bool = True,
    multi_task: bool = False,
    return_weights: bool = False,
    positive_class_weight: float = 1.0,
    batch_reduction: Reduction = "mean"
):
    # Apply softmax if from_logits is True
    if from_logits:
        if y_pred.shape[1] == 1 or multi_task:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
    
    # Handle different input shapes
    if y_pred.shape[1] == 1 or multi_task:
        # Binary classification with [B, 1] output
        # Convert to binary predictions (>= 0.5)
        pred = (y_pred >= 0.5).float()
        correct = pred.eq(y_true.float()).float()
        correct = correct.mean(dim=1)
    else:
        # Multi-class classification with [B, C] output
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


def precision(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute precision for binary classification tasks.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size] or [batch_size, num_tasks]
        y_true: True binary labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid to predictions
        multi_task: Whether to treat as multi-task learning (compute precision per task and average)
    
    Returns:
        Precision score (scalar for single task, averaged across tasks for multi-task)
    """
    # Apply sigmoid if from_logits is True
    if from_logits:
        y_pred = torch.sigmoid(y_pred)
    
    # Convert to binary predictions
    y_hard = (y_pred >= 0.5).to(torch.long)
    
    # Ensure y_true is also long type for consistency
    y_true = y_true.to(torch.long)
    
    if multi_task or (y_pred.dim() > 1 and y_pred.shape[1] > 1):
        # Multi-task case: compute precision for each task and average
        if y_pred.dim() == 1:
            # Single task case, but multi_task=True
            y_hard = y_hard.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        # Compute precision for each task
        eps = torch.finfo(torch.float32).eps
        precision_scores = []
        
        for task_idx in range(y_hard.shape[1]):
            task_pred = y_hard[:, task_idx]
            task_true = y_true[:, task_idx]
            
            tp = ((task_pred == 1) & (task_true == 1)).sum().to(torch.float32)
            fp = ((task_pred == 1) & (task_true == 0)).sum().to(torch.float32)
            
            prec = tp / (tp + fp + eps)
            precision_scores.append(prec)
        
        # Average precision across tasks
        return torch.stack(precision_scores).mean()
    
    else:
        # Single task case: flatten to 1D if needed
        if y_hard.dim() > 1:
            y_hard = y_hard.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        tp = ((y_hard == 1) & (y_true == 1)).sum().to(torch.float32)
        fp = ((y_hard == 1) & (y_true == 0)).sum().to(torch.float32)

        eps = torch.finfo(torch.float32).eps
        prec = tp / (tp + fp + eps)

        return prec


def recall(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute recall for binary classification tasks.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size] or [batch_size, num_tasks]
        y_true: True binary labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid to predictions
        multi_task: Whether to treat as multi-task learning (compute recall per task and average)
    
    Returns:
        Recall score (scalar for single task, averaged across tasks for multi-task)
    """
    # Apply sigmoid if from_logits is True
    if from_logits:
        y_pred = torch.sigmoid(y_pred)
    
    # Convert to binary predictions
    y_hard = (y_pred >= 0.5).to(torch.long)
    
    # Ensure y_true is also long type for consistency
    y_true = y_true.to(torch.long)
    
    if multi_task or (y_pred.dim() > 1 and y_pred.shape[1] > 1):
        # Multi-task case: compute recall for each task and average
        if y_pred.dim() == 1:
            # Single task case, but multi_task=True
            y_hard = y_hard.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        # Compute recall for each task
        eps = torch.finfo(torch.float32).eps
        recall_scores = []
        
        for task_idx in range(y_hard.shape[1]):
            task_pred = y_hard[:, task_idx]
            task_true = y_true[:, task_idx]
            
            tp = ((task_pred == 1) & (task_true == 1)).sum().to(torch.float32)
            fn = ((task_pred == 0) & (task_true == 1)).sum().to(torch.float32)
            
            rec = tp / (tp + fn + eps)
            recall_scores.append(rec)
        
        # Average recall across tasks
        return torch.stack(recall_scores).mean()
    
    else:
        # Single task case: flatten to 1D if needed
        if y_hard.dim() > 1:
            y_hard = y_hard.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        tp = ((y_hard == 1) & (y_true == 1)).sum().to(torch.float32)
        fn = ((y_hard == 0) & (y_true == 1)).sum().to(torch.float32)

        eps = torch.finfo(torch.float32).eps
        rec = tp / (tp + fn + eps)

        return rec


def f1_score(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute F1 score for binary classification tasks.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size] or [batch_size, num_tasks]
        y_true: True binary labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid to predictions
        multi_task: Whether to treat as multi-task learning (compute F1 per task and average)
    
    Returns:
        F1 score (scalar for single task, averaged across tasks for multi-task)
    """
    # Apply sigmoid if from_logits is True
    if from_logits:
        y_pred = torch.sigmoid(y_pred)
    
    # Convert to binary predictions
    y_hard = (y_pred >= 0.5).to(torch.long)
    
    # Ensure y_true is also long type for consistency
    y_true = y_true.to(torch.long)
    
    if multi_task or (y_pred.dim() > 1 and y_pred.shape[1] > 1):
        # Multi-task case: compute F1 for each task and average
        if y_pred.dim() == 1:
            # Single task case, but multi_task=True
            y_hard = y_hard.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        # Compute F1 for each task
        eps = torch.finfo(torch.float32).eps
        f1_scores = []
        
        for task_idx in range(y_hard.shape[1]):
            task_pred = y_hard[:, task_idx]
            task_true = y_true[:, task_idx]
            
            tp = ((task_pred == 1) & (task_true == 1)).sum().to(torch.float32)
            fp = ((task_pred == 1) & (task_true == 0)).sum().to(torch.float32)
            fn = ((task_pred == 0) & (task_true == 1)).sum().to(torch.float32)
            
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = (2.0 * precision * recall) / (precision + recall + eps)
            f1_scores.append(f1)
        
        # Average F1 across tasks
        return torch.stack(f1_scores).mean()
    
    else:
        # Single task case: flatten to 1D if needed
        if y_hard.dim() > 1:
            y_hard = y_hard.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        tp = ((y_hard == 1) & (y_true == 1)).sum().to(torch.float32)
        fp = ((y_hard == 1) & (y_true == 0)).sum().to(torch.float32)
        fn = ((y_hard == 0) & (y_true == 1)).sum().to(torch.float32)

        eps = torch.finfo(torch.float32).eps
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)

        return f1


def auroc(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (AUROC) for binary classification.
    
    AUROC measures the probability that a randomly chosen positive example is ranked higher
    than a randomly chosen negative example. It is threshold-independent.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size] or [batch_size, num_tasks]
        y_true: True binary labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid to predictions
        multi_task: Whether to treat as multi-task learning (compute AUROC per task and average)
    
    Returns:
        AUROC score (scalar for single task, averaged across tasks for multi-task)
    """
    # Apply sigmoid if from_logits is True
    if from_logits:
        y_pred = torch.sigmoid(y_pred)
    
    # Ensure y_true is float type for consistency
    y_true = y_true.to(torch.float32)
    
    if multi_task or (y_pred.dim() > 1 and y_pred.shape[1] > 1):
        # Multi-task case: compute AUROC for each task and average
        if y_pred.dim() == 1:
            # Single task case, but multi_task=True
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        auroc_scores = []
        
        for task_idx in range(y_pred.shape[1]):
            task_pred = y_pred[:, task_idx]
            task_true = y_true[:, task_idx]
            
            # Compute AUROC for this task
            task_auroc = _compute_auroc(task_pred, task_true)
            auroc_scores.append(task_auroc)
        
        # Average AUROC across tasks
        return torch.stack(auroc_scores).mean()
    
    else:
        # Single task case: flatten to 1D if needed
        if y_pred.dim() > 1:
            y_pred = y_pred.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        return _compute_auroc(y_pred, y_true)


def _compute_auroc(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Helper function to compute AUROC for 1D tensors.
    
    Uses the trapezoidal rule to compute the area under the ROC curve.
    """
    # Handle edge cases
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        # All negative or all positive - AUROC is undefined, return 0.5
        return torch.tensor(0.5, device=y_pred.device, dtype=torch.float32)
    
    # Sort predictions in descending order
    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]
    
    # Compute true positives and false positives at each threshold
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    
    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.arange(1, len(y_true) + 1, device=y_pred.device, dtype=torch.float32) - tps
    
    # Compute TPR and FPR
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add (0, 0) point at the beginning
    tpr = torch.cat([torch.tensor([0.0], device=tpr.device, dtype=tpr.dtype), tpr])
    fpr = torch.cat([torch.tensor([0.0], device=fpr.device, dtype=fpr.dtype), fpr])
    
    # Compute area using trapezoidal rule
    auroc = torch.trapz(tpr, fpr)
    
    return auroc


def auprc(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute Area Under the Precision-Recall Curve (AUPRC) for binary classification.
    
    AUPRC is particularly useful for imbalanced datasets where the positive class is rare.
    It measures the area under the curve formed by precision and recall at different thresholds.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size] or [batch_size, num_tasks]
        y_true: True binary labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid to predictions
        multi_task: Whether to treat as multi-task learning (compute AUPRC per task and average)
    
    Returns:
        AUPRC score (scalar for single task, averaged across tasks for multi-task)
    """
    # Apply sigmoid if from_logits is True
    if from_logits:
        y_pred = torch.sigmoid(y_pred)
    
    # Ensure y_true is float type for consistency
    y_true = y_true.to(torch.float32)
    
    if multi_task or (y_pred.dim() > 1 and y_pred.shape[1] > 1):
        # Multi-task case: compute AUPRC for each task and average
        if y_pred.dim() == 1:
            # Single task case, but multi_task=True
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        auprc_scores = []
        
        for task_idx in range(y_pred.shape[1]):
            task_pred = y_pred[:, task_idx]
            task_true = y_true[:, task_idx]
            
            # Compute AUPRC for this task
            task_auprc = _compute_auprc(task_pred, task_true)
            auprc_scores.append(task_auprc)
        
        # Average AUPRC across tasks
        return torch.stack(auprc_scores).mean()
    
    else:
        # Single task case: flatten to 1D if needed
        if y_pred.dim() > 1:
            y_pred = y_pred.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        return _compute_auprc(y_pred, y_true)


def _compute_auprc(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    Helper function to compute AUPRC for 1D tensors.
    
    Uses the trapezoidal rule to compute the area under the precision-recall curve.
    """
    eps = torch.finfo(torch.float32).eps
    
    # Handle edge cases
    if y_true.sum() == 0:
        # All negative - AUPRC is undefined, return 0
        return torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)
    
    if y_true.sum() == len(y_true):
        # All positive - AUPRC is 1
        return torch.tensor(1.0, device=y_pred.device, dtype=torch.float32)
    
    # Sort predictions in descending order
    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]
    
    # Compute true positives and false positives at each threshold
    n_pos = y_true.sum()
    
    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.arange(1, len(y_true) + 1, device=y_pred.device, dtype=torch.float32) - tps
    
    # Compute precision and recall
    precision = tps / (tps + fps + eps)
    recall = tps / (n_pos + eps)
    
    # Add (0, 1) point at the beginning (recall=0, precision=1 or baseline)
    # For PR curve, at recall=0, precision is typically 1 if we predict nothing positive
    # But more accurately, we should use the baseline precision (fraction of positives)
    baseline_precision = n_pos / len(y_true)
    precision = torch.cat([baseline_precision.unsqueeze(0), precision])
    recall = torch.cat([torch.tensor([0.0], device=recall.device, dtype=recall.dtype), recall])
    
    # Compute area using trapezoidal rule
    # Note: For PR curve, we integrate precision with respect to recall
    auprc = torch.trapz(precision, recall)
    
    return auprc
