import inspect
# ionpy imports
from ..experiment.util import eval_config
# torch imports
import torch
from torch import nn


class MultiLoss(nn.Module):
    def __init__(self, losses, weights=None, return_all=True):
        super().__init__()
        if weights is None:
            weights = [1 for _ in losses]
        assert len(weights) == len(losses)
        self.weights = weights
        self.losses = losses
        self.return_all = return_all

        def get_name(loss):
            if inspect.isfunction(loss):
                return loss.__name__
            if inspect.ismethod(loss):
                return loss.__class__.__name__

        self.names = [get_name(loss_func) for loss_func in losses]

    def forward(self, pred, target):
        if not self.return_all:
            return sum(w * fn(pred, target) for w, fn in zip(self.weights, self.losses))
        losses = [(name, fn(pred, target)) for name, fn in zip(self.names, self.losses)]
        losses.append(("all", sum(w * loss for w, loss in zip(self.weights, losses))))
        return dict(losses)


# Define a combined loss function that sums individual losses
class CombinedLoss(nn.Module):

    def __init__(self, loss_func_dict, loss_func_weights):
        super(CombinedLoss, self).__init__()
        self.loss_fn_dict = nn.ModuleDict(loss_func_dict)
        self.loss_func_weights = loss_func_weights

    def forward(self, outputs, targets):
        total_loss = torch.tensor(0.0, device=outputs.device)
        for loss_name, loss_func in self.loss_fn_dict.items():
            total_loss += self.loss_func_weights[loss_name] * loss_func(outputs, targets)
        return total_loss


def eval_combo_config(loss_config):
    # Combined loss functions case
    combo_losses = loss_config["_combo_class"]
    # Instantiate each loss function using eval_config
    loss_fn_dict = {} 
    loss_fn_weights = {} 
    for name, config in combo_losses.items():
        cfg_dict = config.to_dict()
        loss_fn_weights[name] = cfg_dict.pop("weight", 1.0)
        loss_fn_dict[name] = eval_config(cfg_dict)
    # If 'convex_param' is present, return a convex combination of the losses
    if "convex_param" in loss_config:
        # Assert that there are only two losses
        assert len(loss_fn_dict) == 2, "Convex combination of more than two losses is not supported."
        # Assert that the weights are all one (they don't exist)
        for weight in loss_fn_weights.values():
            assert weight == 1.0, "Convex combination of losses with weights is not supported."
        # Assign the weights such that the first loss is weighted by convex_param and the second by 1-convex_param
        convex_param = loss_config["convex_param"]
        for i, name in enumerate(loss_fn_dict.keys()):
            loss_fn_weights[name] = convex_param if i == 0 else (1 - convex_param)

    return CombinedLoss(
        loss_func_dict=loss_fn_dict,
        loss_func_weights=loss_fn_weights
    )