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
    """
    Compute accuracy for binary or multiclass classification.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size], [batch_size, 1], 
                [batch_size, 2], [batch_size, num_tasks], or [batch_size, num_classes]
        y_true: True labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid/softmax to predictions
        multi_task: Whether to treat as multi-task learning (multiple independent binary tasks).
                    If False and y_pred.shape[1] > 2, treats as multiclass classification.
                    Note: [B, 2] is treated as binary (using class 1 probability).
        return_weights: If True and batch_reduction is None, return sample weights
        positive_class_weight: Weight for positive class samples (only used with return_weights)
        batch_reduction: How to reduce over batch ("mean", "sum", or None)
    
    Returns:
        Accuracy score
    """
    # Determine if this is binary, multiclass, or multi-task
    # Binary: [B], [B, 1], or [B, 2] (2-class treated as binary)
    is_binary = y_pred.dim() == 1 or y_pred.shape[1] <= 2
    is_multiclass = not is_binary and not multi_task and y_pred.dim() > 1 and y_pred.shape[1] > 2
    
    if is_multiclass:
        # Multiclass classification (C > 2): apply softmax and take argmax
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        maxk = max((1,))
        _, pred = y_pred.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(y_true.view(1, -1).expand_as(pred)).float().squeeze(0)
    elif multi_task:
        # Multi-task binary classification: apply sigmoid to each task independently
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        pred = (y_pred >= 0.5).float()
        correct = pred.eq(y_true.float()).float()
        correct = correct.mean(dim=1)
    else:
        # Binary classification: [B], [B, 1], or [B, 2]
        if y_pred.dim() > 1 and y_pred.shape[1] == 2:
            # 2-class output: apply softmax and take class 1 probability
            if from_logits:
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred[:, 1]
        else:
            # Single output: apply sigmoid
            if from_logits:
                y_pred = torch.sigmoid(y_pred)
        
        pred = (y_pred >= 0.5).float()
        if pred.dim() > 1:
            pred = pred.view(-1)
        y_true_flat = y_true.view(-1).float() if y_true.dim() > 1 else y_true.float()
        correct = pred.eq(y_true_flat).float()

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
    Compute precision for binary or multiclass classification tasks.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size], [batch_size, 1], 
                [batch_size, 2], [batch_size, num_tasks], or [batch_size, num_classes]
        y_true: True labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid/softmax to predictions
        multi_task: Whether to treat as multi-task learning (multiple independent binary tasks).
                    If False and y_pred.shape[1] > 2, treats as multiclass classification.
                    Note: [B, 2] is treated as binary (using class 1 probability).
    
    Returns:
        Precision score (macro-averaged for multiclass/multi-task)
    """
    eps = torch.finfo(torch.float32).eps
    
    # Ensure y_true is long type for consistency
    y_true = y_true.to(torch.long)
    
    # Determine if this is binary, multiclass, or multi-task
    # Binary: [B], [B, 1], or [B, 2] (2-class treated as binary)
    is_binary = y_pred.dim() == 1 or y_pred.shape[1] <= 2
    is_multiclass = not is_binary and not multi_task and y_pred.dim() > 1 and y_pred.shape[1] > 2
    
    if is_multiclass:
        # Multiclass classification (C > 2): apply softmax and take argmax
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        y_hard = y_pred.argmax(dim=1)
        
        # Flatten y_true if needed
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        # Compute macro-averaged precision across classes
        num_classes = y_pred.shape[1]
        precision_scores = []
        
        for c in range(num_classes):
            tp = ((y_hard == c) & (y_true == c)).sum().to(torch.float32)
            fp = ((y_hard == c) & (y_true != c)).sum().to(torch.float32)
            prec = tp / (tp + fp + eps)
            precision_scores.append(prec)
        
        return torch.stack(precision_scores).mean()
    
    elif multi_task:
        # Multi-task binary classification: apply sigmoid to each task independently
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        y_hard = (y_pred >= 0.5).to(torch.long)
        
        if y_pred.dim() == 1:
            y_hard = y_hard.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        precision_scores = []
        
        for task_idx in range(y_hard.shape[1]):
            task_pred = y_hard[:, task_idx]
            task_true = y_true[:, task_idx]
            
            tp = ((task_pred == 1) & (task_true == 1)).sum().to(torch.float32)
            fp = ((task_pred == 1) & (task_true == 0)).sum().to(torch.float32)
            
            prec = tp / (tp + fp + eps)
            precision_scores.append(prec)
        
        return torch.stack(precision_scores).mean()
    
    else:
        # Binary classification: [B], [B, 1], or [B, 2]
        if y_pred.dim() > 1 and y_pred.shape[1] == 2:
            # 2-class output: apply softmax and take class 1 probability
            if from_logits:
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred[:, 1]
        else:
            # Single output: apply sigmoid
            if from_logits:
                y_pred = torch.sigmoid(y_pred)
        
        y_hard = (y_pred >= 0.5).to(torch.long)
        
        if y_hard.dim() > 1:
            y_hard = y_hard.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        tp = ((y_hard == 1) & (y_true == 1)).sum().to(torch.float32)
        fp = ((y_hard == 1) & (y_true == 0)).sum().to(torch.float32)

        prec = tp / (tp + fp + eps)

        return prec


def recall(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute recall for binary or multiclass classification tasks.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size], [batch_size, 1], 
                [batch_size, 2], [batch_size, num_tasks], or [batch_size, num_classes]
        y_true: True labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid/softmax to predictions
        multi_task: Whether to treat as multi-task learning (multiple independent binary tasks).
                    If False and y_pred.shape[1] > 2, treats as multiclass classification.
                    Note: [B, 2] is treated as binary (using class 1 probability).
    
    Returns:
        Recall score (macro-averaged for multiclass/multi-task)
    """
    eps = torch.finfo(torch.float32).eps
    
    # Ensure y_true is long type for consistency
    y_true = y_true.to(torch.long)
    
    # Determine if this is binary, multiclass, or multi-task
    # Binary: [B], [B, 1], or [B, 2] (2-class treated as binary)
    is_binary = y_pred.dim() == 1 or y_pred.shape[1] <= 2
    is_multiclass = not is_binary and not multi_task and y_pred.dim() > 1 and y_pred.shape[1] > 2
    
    if is_multiclass:
        # Multiclass classification (C > 2): apply softmax and take argmax
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        y_hard = y_pred.argmax(dim=1)
        
        # Flatten y_true if needed
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        # Compute macro-averaged recall across classes
        num_classes = y_pred.shape[1]
        recall_scores = []
        
        for c in range(num_classes):
            tp = ((y_hard == c) & (y_true == c)).sum().to(torch.float32)
            fn = ((y_hard != c) & (y_true == c)).sum().to(torch.float32)
            rec = tp / (tp + fn + eps)
            recall_scores.append(rec)
        
        return torch.stack(recall_scores).mean()
    
    elif multi_task:
        # Multi-task binary classification: apply sigmoid to each task independently
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        y_hard = (y_pred >= 0.5).to(torch.long)
        
        if y_pred.dim() == 1:
            y_hard = y_hard.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        recall_scores = []
        
        for task_idx in range(y_hard.shape[1]):
            task_pred = y_hard[:, task_idx]
            task_true = y_true[:, task_idx]
            
            tp = ((task_pred == 1) & (task_true == 1)).sum().to(torch.float32)
            fn = ((task_pred == 0) & (task_true == 1)).sum().to(torch.float32)
            
            rec = tp / (tp + fn + eps)
            recall_scores.append(rec)
        
        return torch.stack(recall_scores).mean()
    
    else:
        # Binary classification: [B], [B, 1], or [B, 2]
        if y_pred.dim() > 1 and y_pred.shape[1] == 2:
            # 2-class output: apply softmax and take class 1 probability
            if from_logits:
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred[:, 1]
        else:
            # Single output: apply sigmoid
            if from_logits:
                y_pred = torch.sigmoid(y_pred)
        
        y_hard = (y_pred >= 0.5).to(torch.long)
        
        if y_hard.dim() > 1:
            y_hard = y_hard.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        tp = ((y_hard == 1) & (y_true == 1)).sum().to(torch.float32)
        fn = ((y_hard == 0) & (y_true == 1)).sum().to(torch.float32)

        rec = tp / (tp + fn + eps)

        return rec


def f1_score(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute F1 score for binary or multiclass classification tasks.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size], [batch_size, 1], 
                [batch_size, 2], [batch_size, num_tasks], or [batch_size, num_classes]
        y_true: True labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid/softmax to predictions
        multi_task: Whether to treat as multi-task learning (multiple independent binary tasks).
                    If False and y_pred.shape[1] > 2, treats as multiclass classification.
                    Note: [B, 2] is treated as binary (using class 1 probability).
    
    Returns:
        F1 score (macro-averaged for multiclass/multi-task)
    """
    eps = torch.finfo(torch.float32).eps
    
    # Ensure y_true is long type for consistency
    y_true = y_true.to(torch.long)
    
    # Determine if this is binary, multiclass, or multi-task
    # Binary: [B], [B, 1], or [B, 2] (2-class treated as binary)
    is_binary = y_pred.dim() == 1 or y_pred.shape[1] <= 2
    is_multiclass = not is_binary and not multi_task and y_pred.dim() > 1 and y_pred.shape[1] > 2
    
    if is_multiclass:
        # Multiclass classification (C > 2): apply softmax and take argmax
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        y_hard = y_pred.argmax(dim=1)
        
        # Flatten y_true if needed
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        # Compute macro-averaged F1 across classes
        num_classes = y_pred.shape[1]
        f1_scores = []
        
        for c in range(num_classes):
            tp = ((y_hard == c) & (y_true == c)).sum().to(torch.float32)
            fp = ((y_hard == c) & (y_true != c)).sum().to(torch.float32)
            fn = ((y_hard != c) & (y_true == c)).sum().to(torch.float32)
            
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f1 = (2.0 * prec * rec) / (prec + rec + eps)
            f1_scores.append(f1)
        
        return torch.stack(f1_scores).mean()
    
    elif multi_task:
        # Multi-task binary classification: apply sigmoid to each task independently
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        y_hard = (y_pred >= 0.5).to(torch.long)
        
        if y_pred.dim() == 1:
            y_hard = y_hard.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        f1_scores = []
        
        for task_idx in range(y_hard.shape[1]):
            task_pred = y_hard[:, task_idx]
            task_true = y_true[:, task_idx]
            
            tp = ((task_pred == 1) & (task_true == 1)).sum().to(torch.float32)
            fp = ((task_pred == 1) & (task_true == 0)).sum().to(torch.float32)
            fn = ((task_pred == 0) & (task_true == 1)).sum().to(torch.float32)
            
            prec = tp / (tp + fp + eps)
            rec = tp / (tp + fn + eps)
            f1 = (2.0 * prec * rec) / (prec + rec + eps)
            f1_scores.append(f1)
        
        return torch.stack(f1_scores).mean()
    
    else:
        # Binary classification: [B], [B, 1], or [B, 2]
        if y_pred.dim() > 1 and y_pred.shape[1] == 2:
            # 2-class output: apply softmax and take class 1 probability
            if from_logits:
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred[:, 1]
        else:
            # Single output: apply sigmoid
            if from_logits:
                y_pred = torch.sigmoid(y_pred)
        
        y_hard = (y_pred >= 0.5).to(torch.long)
        
        if y_hard.dim() > 1:
            y_hard = y_hard.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        tp = ((y_hard == 1) & (y_true == 1)).sum().to(torch.float32)
        fp = ((y_hard == 1) & (y_true == 0)).sum().to(torch.float32)
        fn = ((y_hard == 0) & (y_true == 1)).sum().to(torch.float32)

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = (2.0 * prec * rec) / (prec + rec + eps)

        return f1


def auroc(
    y_pred: Tensor,
    y_true: Tensor,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (AUROC).
    
    AUROC measures the probability that a randomly chosen positive example is ranked higher
    than a randomly chosen negative example. It is threshold-independent.
    
    For multiclass classification (C > 2), computes one-vs-rest AUROC for each class and returns
    the macro-averaged score.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size], [batch_size, 1], 
                [batch_size, 2], [batch_size, num_tasks], or [batch_size, num_classes]
        y_true: True labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid/softmax to predictions
        multi_task: Whether to treat as multi-task learning (multiple independent binary tasks).
                    If False and y_pred.shape[1] > 2, treats as multiclass classification.
                    Note: [B, 2] is treated as binary (using class 1 probability).
    
    Returns:
        AUROC score (macro-averaged for multiclass/multi-task)
    """
    # Determine if this is binary, multiclass, or multi-task
    # Binary: [B], [B, 1], or [B, 2] (2-class treated as binary)
    is_binary = y_pred.dim() == 1 or y_pred.shape[1] <= 2
    is_multiclass = not is_binary and not multi_task and y_pred.dim() > 1 and y_pred.shape[1] > 2
    
    if is_multiclass:
        # Multiclass classification (C > 2): apply softmax, compute one-vs-rest AUROC
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        
        # Ensure y_true is long type for class indexing
        y_true = y_true.to(torch.long)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        num_classes = y_pred.shape[1]
        auroc_scores = []
        
        for c in range(num_classes):
            # One-vs-rest: probability of class c vs binary label (is class c or not)
            class_pred = y_pred[:, c]
            class_true = (y_true == c).to(torch.float32)
            
            task_auroc = _compute_auroc(class_pred, class_true)
            auroc_scores.append(task_auroc)
        
        return torch.stack(auroc_scores).mean()
    
    elif multi_task:
        # Multi-task binary classification: apply sigmoid to each task independently
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        y_true = y_true.to(torch.float32)
        
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        auroc_scores = []
        
        for task_idx in range(y_pred.shape[1]):
            task_pred = y_pred[:, task_idx]
            task_true = y_true[:, task_idx]
            
            task_auroc = _compute_auroc(task_pred, task_true)
            auroc_scores.append(task_auroc)
        
        return torch.stack(auroc_scores).mean()
    
    else:
        # Binary classification: [B], [B, 1], or [B, 2]
        if y_pred.dim() > 1 and y_pred.shape[1] == 2:
            # 2-class output: apply softmax and take class 1 probability
            if from_logits:
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred[:, 1]
        else:
            # Single output: apply sigmoid
            if from_logits:
                y_pred = torch.sigmoid(y_pred)
        
        y_true = y_true.to(torch.float32)
        
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
    Compute Area Under the Precision-Recall Curve (AUPRC).
    
    AUPRC is particularly useful for imbalanced datasets where the positive class is rare.
    It measures the area under the curve formed by precision and recall at different thresholds.
    
    For multiclass classification (C > 2), computes one-vs-rest AUPRC for each class and returns
    the macro-averaged score.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size], [batch_size, 1], 
                [batch_size, 2], [batch_size, num_tasks], or [batch_size, num_classes]
        y_true: True labels of shape [batch_size] or [batch_size, num_tasks]
        from_logits: Whether to apply sigmoid/softmax to predictions
        multi_task: Whether to treat as multi-task learning (multiple independent binary tasks).
                    If False and y_pred.shape[1] > 2, treats as multiclass classification.
                    Note: [B, 2] is treated as binary (using class 1 probability).
    
    Returns:
        AUPRC score (macro-averaged for multiclass/multi-task)
    """
    # Determine if this is binary, multiclass, or multi-task
    # Binary: [B], [B, 1], or [B, 2] (2-class treated as binary)
    is_binary = y_pred.dim() == 1 or y_pred.shape[1] <= 2
    is_multiclass = not is_binary and not multi_task and y_pred.dim() > 1 and y_pred.shape[1] > 2
    
    if is_multiclass:
        # Multiclass classification (C > 2): apply softmax, compute one-vs-rest AUPRC
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        
        # Ensure y_true is long type for class indexing
        y_true = y_true.to(torch.long)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        num_classes = y_pred.shape[1]
        auprc_scores = []
        
        for c in range(num_classes):
            # One-vs-rest: probability of class c vs binary label (is class c or not)
            class_pred = y_pred[:, c]
            class_true = (y_true == c).to(torch.float32)
            
            task_auprc = _compute_auprc(class_pred, class_true)
            auprc_scores.append(task_auprc)
        
        return torch.stack(auprc_scores).mean()
    
    elif multi_task:
        # Multi-task binary classification: apply sigmoid to each task independently
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        y_true = y_true.to(torch.float32)
        
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        auprc_scores = []
        
        for task_idx in range(y_pred.shape[1]):
            task_pred = y_pred[:, task_idx]
            task_true = y_true[:, task_idx]
            
            task_auprc = _compute_auprc(task_pred, task_true)
            auprc_scores.append(task_auprc)
        
        return torch.stack(auprc_scores).mean()
    
    else:
        # Binary classification: [B], [B, 1], or [B, 2]
        if y_pred.dim() > 1 and y_pred.shape[1] == 2:
            # 2-class output: apply softmax and take class 1 probability
            if from_logits:
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred[:, 1]
        else:
            # Single output: apply sigmoid
            if from_logits:
                y_pred = torch.sigmoid(y_pred)
        
        y_true = y_true.to(torch.float32)
        
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


def prec_at_recall(
    y_pred: Tensor,
    y_true: Tensor,
    min_recall: float = 0.9,
    from_logits: bool = True,
    multi_task: bool = False,
):
    """
    Compute precision at a threshold where recall is at least min_recall.
    
    This is useful when you want to ensure a minimum recall level and want to 
    know the corresponding precision at that operating point.
    
    For multiclass classification (C > 2), computes one-vs-rest precision at recall for each 
    class and returns the macro-averaged score.
    
    Args:
        y_pred: Predicted probabilities/logits of shape [batch_size], [batch_size, 1], 
                [batch_size, 2], [batch_size, num_tasks], or [batch_size, num_classes]
        y_true: True labels of shape [batch_size] or [batch_size, num_tasks]
        min_recall: Minimum recall threshold (default: 0.9). The function finds the 
                    operating point where recall >= min_recall and returns precision there.
        from_logits: Whether to apply sigmoid/softmax to predictions
        multi_task: Whether to treat as multi-task learning (multiple independent binary tasks).
                    If False and y_pred.shape[1] > 2, treats as multiclass classification.
                    Note: [B, 2] is treated as binary (using class 1 probability).
    
    Returns:
        Precision score at the threshold where recall >= min_recall.
        If recall cannot reach min_recall, returns the precision at max recall.
        (macro-averaged for multiclass/multi-task)
    """
    # Determine if this is binary, multiclass, or multi-task
    # Binary: [B], [B, 1], or [B, 2] (2-class treated as binary)
    is_binary = y_pred.dim() == 1 or y_pred.shape[1] <= 2
    is_multiclass = not is_binary and not multi_task and y_pred.dim() > 1 and y_pred.shape[1] > 2
    
    if is_multiclass:
        # Multiclass classification (C > 2): apply softmax, compute one-vs-rest prec@recall
        if from_logits:
            y_pred = torch.softmax(y_pred, dim=1)
        
        # Ensure y_true is long type for class indexing
        y_true = y_true.to(torch.long)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        num_classes = y_pred.shape[1]
        prec_scores = []
        
        for c in range(num_classes):
            # One-vs-rest: probability of class c vs binary label (is class c or not)
            class_pred = y_pred[:, c]
            class_true = (y_true == c).to(torch.float32)
            
            task_prec = _compute_prec_at_recall(class_pred, class_true, min_recall)
            prec_scores.append(task_prec)
        
        return torch.stack(prec_scores).mean()
    
    elif multi_task:
        # Multi-task binary classification: apply sigmoid to each task independently
        if from_logits:
            y_pred = torch.sigmoid(y_pred)
        
        y_true = y_true.to(torch.float32)
        
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        
        prec_scores = []
        
        for task_idx in range(y_pred.shape[1]):
            task_pred = y_pred[:, task_idx]
            task_true = y_true[:, task_idx]
            
            task_prec = _compute_prec_at_recall(task_pred, task_true, min_recall)
            prec_scores.append(task_prec)
        
        return torch.stack(prec_scores).mean()
    
    else:
        # Binary classification: [B], [B, 1], or [B, 2]
        if y_pred.dim() > 1 and y_pred.shape[1] == 2:
            # 2-class output: apply softmax and take class 1 probability
            if from_logits:
                y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred[:, 1]
        else:
            # Single output: apply sigmoid
            if from_logits:
                y_pred = torch.sigmoid(y_pred)
        
        y_true = y_true.to(torch.float32)
        
        if y_pred.dim() > 1:
            y_pred = y_pred.view(-1)
        if y_true.dim() > 1:
            y_true = y_true.view(-1)
        
        return _compute_prec_at_recall(y_pred, y_true, min_recall)


def _compute_prec_at_recall(y_pred: Tensor, y_true: Tensor, min_recall: float) -> Tensor:
    """
    Helper function to compute precision at a given recall threshold for 1D tensors.
    
    Finds the threshold where recall >= min_recall and returns the precision at that point.
    """
    eps = torch.finfo(torch.float32).eps
    
    # Handle edge cases
    n_pos = y_true.sum()
    
    if n_pos == 0:
        # No positives - precision is undefined, return 0
        return torch.tensor(0.0, device=y_pred.device, dtype=torch.float32)
    
    if n_pos == len(y_true):
        # All positive - at any threshold, recall = precision = 1
        return torch.tensor(1.0, device=y_pred.device, dtype=torch.float32)
    
    # Sort predictions in descending order
    sorted_indices = torch.argsort(y_pred, descending=True)
    y_true_sorted = y_true[sorted_indices]
    
    # Compute true positives and false positives at each threshold
    tps = torch.cumsum(y_true_sorted, dim=0)
    fps = torch.arange(1, len(y_true) + 1, device=y_pred.device, dtype=torch.float32) - tps
    
    # Compute precision and recall at each threshold
    precision_vals = tps / (tps + fps + eps)
    recall_vals = tps / (n_pos + eps)
    
    # Find the first index where recall >= min_recall
    # We want the smallest threshold (highest index in sorted order) that achieves min_recall
    # because that gives us the best precision while meeting the recall constraint
    recall_mask = recall_vals >= min_recall
    
    if not recall_mask.any():
        # Cannot achieve min_recall, return precision at max recall (last position)
        return precision_vals[-1]
    
    # Find the first index where recall >= min_recall
    # This corresponds to the highest threshold that achieves the desired recall
    valid_idx = torch.argmax(recall_mask.to(torch.int32))
    
    return precision_vals[valid_idx]
