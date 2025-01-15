# Misc imports
import os
import sys
import time
import copy
import pathlib
from typing import List
from typing import Optional
# Torch imports
import torch
from torch import nn
from torch.amp import GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
# Local imports
from ..nn.util import num_params, split_param_groups_by_weight_decay
from ..util.ioutil import autosave
from ..util.meter import MeterDict
from ..util.torchutils import to_device
from .base import BaseExperiment
from .util import absolute_import, eval_config


class TrainExperiment(BaseExperiment):

    def __init__(
        self, 
        path, 
        set_seed=True, 
        init_metrics=True, 
        load_data=True, 
        load_aug_pipeline=True
    ):
        torch.backends.cudnn.benchmark = True
        super().__init__(path, set_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.build_optim()
        self.build_augmentations(load_aug_pipeline)
        self.build_metrics(init_metrics)
        self.build_data(load_data)
        self.build_loss()


    def build_data(self, load_data):
        data_cfg = self.config["data"].to_dict()
        dataset_constructor = data_cfg.pop("_class", None) or data_cfg.pop("_fn")
        # Get the train and val transforms
        train_transforms = data_cfg.pop("train_transforms", None) 
        val_transforms = data_cfg.pop("val_transforms", None)
        dataset_cls = absolute_import(dataset_constructor)
        if load_data:
            self.train_dataset = dataset_cls(
                split="train", transforms=train_transforms, **data_cfg
            )
            self.val_dataset = dataset_cls(
                split="val", transforms=val_transforms, **data_cfg
            )

    def build_dataloader(self):
        assert self.config["dataloader.batch_size"] <= len(
            self.train_dataset
        ), "Batch size larger than dataset"
        dl_cfg = self.config["dataloader"]
        self.train_dl = DataLoader(
            self.train_dataset, shuffle=True, drop_last=False, **dl_cfg
        )
        self.val_dl = DataLoader(
            self.val_dataset, shuffle=False, drop_last=False, **dl_cfg
        )

    def build_model(self, compile_model=False):
        # Peek at the model class, if it is a Timm model then we need to 
        self.model = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)
        # Used from MultiverSeg Code.
        if torch.__version__ >= "2.0.0" and compile_model:
            self.model = torch.compile(self.model)
            self.compiled = True
        else:
            self.compiled = False
        # Move the model to the device
        self.to_device()

    def build_optim(self):
        optim_cfg = self.config["optim"].to_dict()

        if 'lr_scheduler' in optim_cfg:
            self.lr_scheduler = eval_config(optim_cfg.pop('lr_scheduler', None))

        if "weight_decay" in optim_cfg:
            optim_cfg["params"] = split_param_groups_by_weight_decay(
                self.model, optim_cfg["weight_decay"]
            )
        else:
            optim_cfg["params"] = self.model.parameters()

        self.optim = eval_config(optim_cfg)
        # Zero out the gradients as initialization 
        self.optim.zero_grad()

    def build_loss(self):
        self.loss_func = eval_config(self.config["loss_func"])

    def build_metrics(self, init_metrics):
        self.metric_fns = {}
        if init_metrics:
            if "log.metrics" in self.config:
                self.metric_fns = eval_config(copy.deepcopy(self.config["log.metrics"]))

    def build_initialization(self):
        if "initialization" in self.config:
            init_cfg = self.config["initialization"].to_dict()
            path = pathlib.Path(init_cfg["path"])
            with path.open("rb") as f:
                state = torch.load(f, map_location=self.device)
            if not init_cfg.get("optim", True):
                state.pop("optim", None)
            strict = init_cfg.get("strict", True)
            self.set_state(state, strict=strict)
            print(f"Loaded initialization state from: {path}")

    @property
    def state(self):
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "_epoch": self.properties["epoch"],
        }

    def set_state(self, state, strict=True):
        for attr, state_dict in state.items():
            if not attr.startswith("_"):
                x = getattr(self, attr)
                if isinstance(x, nn.Module):
                    x.load_state_dict(state_dict, strict=strict)
                elif isinstance(x, torch.optim.Optimizer):
                    x.load_state_dict(state_dict)
                else:
                    raise TypeError(f"Unsupported type {type(x)}")

        self._checkpoint_epoch = state["_epoch"]

    def checkpoint(self, tag=None):
        self.properties["epoch"] = self._epoch

        checkpoint_dir = self.path / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        tag = tag if tag is not None else "last"
        print(f"Checkpointing with tag:{tag} at epoch:{self._epoch}")

        with (checkpoint_dir / f"{tag}.pt").open("wb") as f:
            torch.save(self.state, f)

    @property
    def checkpoints(self, as_paths=False) -> List[str]:
        checkpoints = list((self.path / "checkpoints").iterdir())
        checkpoints = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
        if as_paths:
            return checkpoints
        return [c.stem for c in checkpoints]

    def load(self, tag=None, verbose=False):
        checkpoint_dir = self.path / "checkpoints"
        tag = tag if tag is not None else "last"
        with (checkpoint_dir / f"{tag}.pt").open("rb") as f:
            state = torch.load(f, map_location=self.device, weights_only=True)
            self.set_state(state)
        if verbose:
            print(
                f"Loaded checkpoint with tag:{tag}. Last epoch:{self.properties['epoch']}"
            )
        return self

    def to_device(self):
        self.model = to_device(
            self.model, self.device, self.config.get("train.channels_last", False)
        )

    def run_callbacks(self, callback_group, **kwargs):
        for callback in self.callbacks.get(callback_group, []):
            callback(**kwargs)

    def run(
        self,
    ):
        print(f"Running {str(self)}")
        epochs: int = self.config["train.epochs"]

        # If using mixed precision, then create a GradScaler to scale gradients during mixed precision training.
        if self.config.get('experiment.torch_mixed_precision', False):
            self.grad_scaler = GradScaler('cuda')

        self.build_dataloader()
        self.build_callbacks()

        last_epoch: int = self.properties.get("epoch", -1)
        if last_epoch >= 0:
            self.load(tag="last")
            df = self.metrics.df
            autosave(df[df.epoch < last_epoch], self.path / "metrics.jsonl")
        else:
            self.build_initialization()

        checkpoint_freq: int = self.config.get("log.checkpoint_freq", 1)
        eval_freq: int = self.config.get("train.eval_freq", 1)

        for epoch in range(last_epoch + 1, epochs):
            self._epoch = epoch

            # Either we run a validation epoch first and then do a round of training...
            if not self.config['experiment'].get('val_first', False):
                print(f"Start training epoch {epoch}.")
                self.run_phase("train", epoch)

            # Evaluate the model on the validation set.
            if eval_freq > 0 and (epoch % eval_freq == 0 or epoch == epochs - 1):
                print(f"Start validation round at {epoch}.")
                self.run_phase("val", epoch)

            # ... or we run a training epoch first and then do a round of validation.
            if self.config['experiment'].get('val_first', False):
                print(f"Start training epoch {epoch}.")
                self.run_phase("train", epoch)

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.checkpoint()

            self.run_callbacks("epoch", epoch=epoch)

        self.checkpoint(tag="last")
        self.run_callbacks("wrapup")

    def run_phase(self, phase, epoch):
        dl = getattr(self, f"{phase}_dl")
        grad_enabled = phase == "train"

        # Check if augmentations are enabled
        aug_cfg = None if "augmentations" not in self.config else self.config["augmentations"] 
        augmentation = (phase == "train") and (aug_cfg is not None)

        self.model.train(grad_enabled)  # For dropout, batchnorm, &c

        phase_meters = MeterDict()
        iter_loader = iter(dl)

        with torch.set_grad_enabled(grad_enabled):
            for batch_idx in range(len(dl)):
                batch = next(iter_loader) # Doing this lets us time the data loading.

                outputs = self.run_step(
                    batch_idx=batch_idx,
                    batch=batch,
                    backward=grad_enabled,
                    augmentation=augmentation,
                    epoch=epoch,
                    phase=phase,
                )
                
                batch_metrics, batch_metric_weights = self.compute_metrics(outputs)

                phase_meters.update(batch_metrics, weights=batch_metric_weights)

                self.run_callbacks(
                    "batch", 
                    epoch=epoch, 
                    batch_idx=batch_idx, 
                    phase=phase
                )
        phase_metrics = {"phase": phase, "epoch": epoch, **phase_meters.collect("mean")}

        self.metrics.log(phase_metrics)

        return phase_metrics

    def run_step(
        self, 
        batch_idx, 
        batch, 
        epoch=None, 
        phase=None,
        backward=True, 
        augmentation=True, 
    ):
        batch = to_device(
            batch, self.device, self.config.get("train.channels_last", False)
        )

        x, y = batch["img"], batch["label"]

        y_hat = self.model(x)

        loss = self.loss_func(y_hat, y)

        if backward:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        
        forward_batch = {
            "loss": loss, "x": x, "y_true": y, "y_pred": y_hat
        }
        # Run step-wise callbacks if you have them.
        self.run_callbacks("step", batch=forward_batch)

        return forward_batch

    def compute_metrics(self, outputs):
        metrics = {"loss": outputs["loss"].item()}
        metric_weights = {"loss": None}
        for name, fn in self.metric_fns.items():
            value_obj = fn(outputs["y_pred"], outputs["y_true"])
            if isinstance(value_obj, tuple):
                # Place both the values and the weights from the tuple. Note that
                # that if we supply the weights it has to be the unreduced loss.
                value, weight = value_obj
                metrics[name] = value.tolist() 
                metric_weights[name] = weight.tolist()
            else:
                # If the value is a Tensor AND is 1-dimensional, then we can convert it to a float.
                if isinstance(value_obj, torch.Tensor):
                    value_obj = value_obj.item()
                # Place the reduced value_obj into the metrics dictionary, None for metric
                # weight as we haven't returned per-sample weights.
                metrics[name] = value_obj
                metric_weights[name] = None
        return metrics, metric_weights

    def build_augmentations(self, load_aug_pipeline):
        pass