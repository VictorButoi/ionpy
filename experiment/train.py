# Misc imports
import time
import copy
import pathlib
from typing import List
from pprint import pprint
# Torch imports
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
# Local imports
from .base import BaseExperiment
from ..util.ioutil import autosave
from ..util.meter import MeterDict
from ..util.torchutils import to_device
from .util import absolute_import, eval_config
from ..nn.ema import EMAWrapper
from ..nn.util import num_params, split_param_groups_by_weight_decay


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
        dataset_cls = absolute_import(dataset_constructor)
        # Get specific dataset arguments if they exist.
        train_data_kwargs = data_cfg.pop("train_kwargs", {})
        val_data_kwargs = data_cfg.pop("train_kwargs", {})
        # Get the train and val transforms.
        train_transforms = data_cfg.pop("train_transforms", None) 
        val_transforms = data_cfg.pop("val_transforms", None)
        if load_data:
            self.train_dataset = dataset_cls(
                split="train", 
                transforms=train_transforms, 
                **data_cfg,
                **train_data_kwargs
            )
            self.val_dataset = dataset_cls(
                split="val", 
                transforms=val_transforms, 
                **data_cfg,
                **val_data_kwargs
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
        self.model = eval_config(self.config["model"])
        self.properties["num_params"] = num_params(self.model)
        print("Number of parameters:", self.properties["num_params"])
        # Used from MultiverSeg Code.
        if torch.__version__ >= "2.0.0" and compile_model:
            self.model = torch.compile(self.model)
            self.compiled = True
        else:
            self.compiled = False
        
        # If the pretrained_dir exists, then load the model from the directory.
        pretrained_dir = self.config["train"].get("pretrained_dir", None)
        if pretrained_dir is not None:
            path = pathlib.Path(pretrained_dir)
            chkpt_name = self.config["train"].get("load_chkpt", "last")
            weights_path = path / "checkpoints" / f"{chkpt_name}.pt"
            with weights_path.open("rb") as f:
                state = torch.load(f, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state["model"])
            print(f"Loaded model from: {weights_path}")
        # If the model has an EMA wrapper, then we need to wrap it.
        if 'ema' in self.config:
            self.model = EMAWrapper(
                self.model,
                **self.config['ema'].to_dict()
            ) 
        # Move the model to the device
        self.to_device()

    def build_optim(self):
        optim_cfg = self.config["optim"].to_dict()

        # First we need to get the optimizer schduler.
        lr_scheduler_cfg = optim_cfg.pop('lr_scheduler', None)

        if "weight_decay" in optim_cfg:
            optim_cfg["params"] = split_param_groups_by_weight_decay(
                self.model, optim_cfg["weight_decay"]
            )
        else:
            optim_cfg["params"] = self.model.parameters()
        self.optim = eval_config(optim_cfg)

        # If our scheduler is not none, then we need to set it up here.
        if lr_scheduler_cfg is not None:
            self.lr_scheduler = absolute_import(lr_scheduler_cfg.pop("_class"))(
                self.optim, T_max=self.config["train"]["epochs"], **lr_scheduler_cfg
            )
        else:
            self.lr_scheduler = None

        # If the pretrained_dir exists, then load the optimizer state 
        # dict.
        pretrained_dir = self.config["train"].get("pretrained_dir", None)
        if pretrained_dir is not None:
            path = pathlib.Path(pretrained_dir)
            chkpt_name = self.config["train"].get("load_chkpt", "last")
            weights_path = path / "checkpoints" / f"{chkpt_name}.pt"
            with weights_path.open("rb") as f:
                state = torch.load(f, map_location=self.device, weights_only=True)
            self.optim.load_state_dict(state["optim"])
            print(f"Loaded optimizer from: {weights_path}")
        else:
            # Zero out the gradients as initialization 
            self.optim.zero_grad()

    def build_loss(self):
        self.loss_func = eval_config(self.config["loss_func"])
        # If loss_func is a class, then instantiate it.
        if isinstance(self.loss_func, type):
            self.loss_func = self.loss_func()

    def build_metrics(self, init_metrics):
        self.metric_fns = {}
        if init_metrics:
            if "log.metrics" in self.config:
                log_metric_cfg = self.config["log.metrics"].to_dict()
                # Get out the label types if they exist.
                self.metric_label_types = {}
                for metric_name, metric_dict in log_metric_cfg.items():
                    self.metric_label_types[metric_name] = metric_dict.pop("label_type", None)
                # Initialize the metric functions.
                self.metric_fns = eval_config(log_metric_cfg)

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

    def run(self):
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

        # Either we run a validation epoch first and then do a round of training...
        if not self.config['experiment'].get('val_first', False):
            epoch_order = ["train", "val"]
        else:
            # Or we run a training epoch first and then do a round of validation...
            epoch_order = ["val", "train"]

        # Go through epochs of training.
        for epoch in range(last_epoch + 1, epochs):
            self._epoch = epoch
            
            # Do a round of phases.
            for eo in epoch_order:
                print(f"Start {eo} epoch {epoch}.")
                if eo == "train" or eval_freq > 0 and (epoch % eval_freq == 0 or epoch == epochs - 1):
                    self.run_phase(eo, epoch) 

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.checkpoint()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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
                # # We want to time each part of our pipeline to see where the bottleneck is.
                # torch.cuda.synchronize()
                # t1 = time.time()
                batch = next(iter_loader) # Doing this lets us time the data loading.
                # torch.cuda.synchronize()
                # t2 = time.time()
                # print("Data loading time:", t2 - t1)

                # torch.cuda.synchronize()
                # t1 = time.time()
                outputs = self.run_step(
                    batch_idx=batch_idx,
                    batch=batch,
                    backward=grad_enabled,
                    augmentation=augmentation,
                    epoch=epoch,
                    phase=phase,
                )
                # torch.cuda.synchronize()
                # t2 = time.time()
                # print("Forward pass + backwards pass time:", t2 - t1)
                
                # torch.cuda.synchronize()
                # t1 = time.time()
                batch_metrics, batch_metric_weights = self.compute_metrics(outputs)
                # torch.cuda.synchronize()
                # t2 = time.time()
                # print("Compute Metrics time:", t2 - t1)
                # print()

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
        batch_loss = outputs["loss"]
        # If the loss is not a scalar, then we need to reduce it to a scalar.
        if len(batch_loss.shape) != 0:
            batch_loss = batch_loss.mean()
        metrics = {"loss": batch_loss.item()}
        metric_weights = {"loss": None}
        for name, fn in self.metric_fns.items():
            y_pred = outputs["y_pred"]
            y_true = outputs["y_true"]
            # If y_pred and y_true are dictionaries, then we need to choose the correct
            # key for the metric function.
            if isinstance(y_pred, dict) and isinstance(y_true, dict):
                label_type = self.metric_label_types[name]
                y_pred = y_pred[label_type]
                y_true = y_true[label_type]
            # Run the outputs through the metric function.
            # Sometimes we need out outputs to be weighted, so we will
            # return both values and weights.
            value_obj = fn(y_pred, y_true)

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