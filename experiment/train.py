# Misc imports
import os
import pathlib
import numpy as np
from typing import List
import time
import inspect
import warnings
from collections import defaultdict
# Torch imports
import torch
from torch import nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
# Local imports
from .base import BaseExperiment
from ..util.ioutil import autosave
from ..util.meter import MeterDict
from ..util.torchutils import to_device
from .util import absolute_import, eval_config, load_model_from_path, load_optim_from_path
from ..nn.ema import EMAWrapper
from ..nn.util import num_params, split_param_groups_by_weight_decay


def _materialize_scalar_metrics(metrics):
    materialized = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ValueError(
                    f"Metric {key!r} must be a scalar tensor at log time, got shape {tuple(value.shape)}."
                )
            value = value.item()
        materialized[key] = value
    return materialized


class TrainExperiment(BaseExperiment):

    def __init__(
        self, 
        path, 
        set_seed=True, 
        init_metrics=True, 
        load_data=True, 
        load_optim=True,
        load_aug_pipeline=True
    ):
        torch.backends.cudnn.benchmark = True
        self.lr_scheduler = None
        self.lr_scheduler_interval = "epoch"
        self._scheduler_steps_per_epoch = None
        self._global_step = 0
        super().__init__(path, set_seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()
        self.build_augmentations(load_aug_pipeline)
        self.build_metrics(init_metrics)
        self.build_loss()
        self.build_data(load_data)
        self.build_optim(load_optim)
        self.build_callbacks()

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
            self.build_dataloader()

    def build_dataloader(self):
        if self.config["dataloader.batch_size"] > len(self.train_dataset):
            print("Warning: Batch size larger than dataset")
        dl_cfg = self.config["dataloader"]
        self.train_dl = DataLoader(
            self.train_dataset, shuffle=True, drop_last=False, **dl_cfg
        )
        self.val_dl = DataLoader(
            self.val_dataset, shuffle=False, drop_last=False, **dl_cfg
        )

    @staticmethod
    def _maybe_compile(module, compile_cfg):
        """Apply torch.compile if compile_cfg is provided and enabled."""
        if compile_cfg is None:
            return module
        if hasattr(compile_cfg, 'to_dict'):
            compile_cfg = compile_cfg.to_dict()
        else:
            compile_cfg = dict(compile_cfg)
        if not compile_cfg.pop('enabled'):
            return module
        print(f"Compiling {module.__class__.__name__} with torch.compile({compile_cfg})")
        return torch.compile(module, **compile_cfg)

    def build_model(self):
        total_config = self.config.to_dict()
        model_cfg = total_config["model"]
        pretrained_cfg = total_config.get("pretrained", {})
        ema_kwargs = model_cfg.pop("ema", {})
        verbose = model_cfg.pop("verbose", False)
        compile_cfg = model_cfg.pop("compile_cfg", None)
        # Build the model.
        self.model = eval_config(model_cfg)
        self.properties["num_params"] = num_params(self.model)
        print("Main model #params: {:,}".format(self.properties["num_params"]))
        # If the model has an EMA wrapper, then we need to wrap it.
        if ema_kwargs != {}:
            self.model = EMAWrapper(
                self.model,
                **ema_kwargs
            ) 
        # Load pretrained weights BEFORE compiling so state_dict keys match.
        if pretrained_cfg != {}:
            load_model_from_path(
                self.model, 
                device=self.device,
                **pretrained_cfg,
            )
        # Compile the model if compile_cfg is provided and enabled.
        self.model = self._maybe_compile(self.model, compile_cfg)
        # Move the model to the device
        self.to_device()
        # Print the model architecture for debugging (e.g., batchnorm inspection)
        if verbose:
            print("=" * 60)
            print("MODEL ARCHITECTURE:")
            print("=" * 60)
            print(self.model)
            print("=" * 60)

    def build_optim(self, load_optim):
        if load_optim:
            optim_cfg = self.config["optim"].to_dict()
            model_cfg = self.config["model"].to_dict()
            pt_kwargs = model_cfg.pop("pt_kwargs", {})
            # First we need to get the optimizer schduler.
            lr_scheduler_cfg = optim_cfg.pop('lr_scheduler', None)
            warmup_cfg = optim_cfg.pop("warmup", None)

            if "weight_decay" in optim_cfg:
                optim_cfg["params"] = split_param_groups_by_weight_decay(
                    self.model, optim_cfg["weight_decay"]
                )
            else:
                # Only include parameters that require gradients (not frozen)
                optim_cfg["params"] = [p for p in self.model.parameters() if p.requires_grad]
            self.optim = eval_config(optim_cfg)

            self.lr_scheduler = self._create_lr_scheduler(lr_scheduler_cfg, warmup_cfg)

            # If the pretrained_dir exists, then load the optimizer state dict.
            if pt_kwargs != {}:
                load_optim_from_path(
                    self.optim, 
                    device=self.device,
                    **pt_kwargs,
                )
            else:
                # Zero out the gradients as initialization 
                self.optim.zero_grad()

    def _resolve_steps_per_epoch(self):
        if not hasattr(self, "train_dl") or self.train_dl is None:
            return None
        return len(self.train_dl)

    def _parse_lr_scheduler_config(self, lr_scheduler_cfg):
        if lr_scheduler_cfg is None:
            return None, {}, "epoch"
        if hasattr(lr_scheduler_cfg, "to_dict"):
            lr_scheduler_cfg = lr_scheduler_cfg.to_dict()

        if isinstance(lr_scheduler_cfg, str):
            return lr_scheduler_cfg, {}, "epoch"
        if isinstance(lr_scheduler_cfg, dict):
            scheduler_kwargs = dict(lr_scheduler_cfg)
            interval = scheduler_kwargs.pop("interval", "epoch")
            scheduler_path = scheduler_kwargs.pop("_class", None)
            return scheduler_path, scheduler_kwargs, interval
        raise TypeError(
            "optim.lr_scheduler must be either None, a class-path string, "
            f"or a config dict; got {type(lr_scheduler_cfg).__name__}."
        )

    def _normalize_scheduler_interval(self, interval):
        if interval is None:
            return "epoch"
        if interval not in {"epoch", "step"}:
            raise ValueError(
                "optim.lr_scheduler.interval must be either 'epoch' or 'step'; "
                f"got {interval!r}."
            )
        return interval

    def _resolve_total_scheduler_steps(self, interval, steps_per_epoch):
        epochs = int(self.config["train"]["epochs"])
        if interval == "epoch":
            return epochs
        if steps_per_epoch is None:
            raise ValueError(
                "optim.lr_scheduler.interval='step' requires an initialized train dataloader."
            )
        return epochs * steps_per_epoch

    def _resolve_warmup_iters(self, warmup_cfg, interval, steps_per_epoch):
        if warmup_cfg is None:
            return None
        if hasattr(warmup_cfg, "to_dict"):
            warmup_cfg = warmup_cfg.to_dict()
        warmup_cfg = dict(warmup_cfg)

        has_num_epochs = "num_epochs" in warmup_cfg
        has_num_steps = "num_steps" in warmup_cfg
        if has_num_epochs == has_num_steps:
            raise ValueError(
                "optim.warmup must define exactly one of 'num_epochs' or 'num_steps'."
            )

        if interval == "epoch":
            if has_num_steps:
                raise ValueError(
                    "optim.warmup.num_steps is only supported when "
                    "optim.lr_scheduler.interval='step'."
                )
            total_iters = int(warmup_cfg["num_epochs"])
        else:
            if has_num_steps:
                total_iters = int(warmup_cfg["num_steps"])
            else:
                if steps_per_epoch is None:
                    raise ValueError(
                        "optim.warmup.num_epochs requires an initialized train dataloader "
                        "when optim.lr_scheduler.interval='step'."
                    )
                total_iters = int(warmup_cfg["num_epochs"]) * steps_per_epoch

        if total_iters <= 0:
            raise ValueError(f"optim.warmup must resolve to a positive duration, got {total_iters}.")

        return total_iters, warmup_cfg

    def _create_lr_scheduler(self, lr_scheduler_cfg, warmup_cfg=None):
        scheduler_path, scheduler_kwargs, interval = self._parse_lr_scheduler_config(lr_scheduler_cfg)
        interval = self._normalize_scheduler_interval(interval)
        steps_per_epoch = self._resolve_steps_per_epoch()
        self.lr_scheduler_interval = interval
        self._scheduler_steps_per_epoch = steps_per_epoch

        if scheduler_path is None:
            return None

        scheduler_cls = absolute_import(scheduler_path)
        if inspect.isclass(scheduler_cls) and issubclass(
            scheduler_cls,
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            raise ValueError(
                "optim.lr_scheduler does not support metric-driven schedulers such as "
                "ReduceLROnPlateau in this trainer yet."
            )

        scheduler_signature = inspect.signature(scheduler_cls)
        if "T_max" in scheduler_signature.parameters and "T_max" not in scheduler_kwargs:
            scheduler_kwargs["T_max"] = self._resolve_total_scheduler_steps(interval, steps_per_epoch)

        main_scheduler = scheduler_cls(self.optim, **scheduler_kwargs)

        warmup_spec = self._resolve_warmup_iters(warmup_cfg, interval, steps_per_epoch)
        if warmup_spec is None:
            return main_scheduler

        warmup_iters, warmup_cfg = warmup_spec
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optim,
            start_factor=warmup_cfg.get("start_factor", 0.01),
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            self.optim,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_iters],
        )

    def build_loss(self):
        self.loss_func = eval_config(self.config["loss_func"])
        # If loss_func is a class, then instantiate it.
        if isinstance(self.loss_func, type):
            self.loss_func = self.loss_func()

    def build_metrics(self, init_metrics):
        self.metric_fns = {}
        self.global_metric_fns = {}
        if init_metrics:
            if "log.metrics" in self.config:
                self.metric_fns = eval_config(
                    self.config["log.metrics"].to_dict()
                )
            global_metrics_cfg = self.config.get("log.global_metrics", None)
            if global_metrics_cfg is not None:
                self.global_metric_fns = eval_config(
                    global_metrics_cfg.to_dict()
                )

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
            "_lr_scheduler_state": None if self.lr_scheduler is None else self.lr_scheduler.state_dict(),
            "_epoch": self.properties["epoch"],
            "_global_step": self._global_step,
        }

    def set_state(self, state, strict=True):
        scheduler_state = state.get("_lr_scheduler_state")
        checkpoint_epoch = state["_epoch"]
        checkpoint_global_step = state.get("_global_step")

        for attr, state_dict in state.items():
            if attr == "optim":
                continue
            if not attr.startswith("_"):
                x = getattr(self, attr)
                if isinstance(x, nn.Module):
                    x.load_state_dict(state_dict, strict=strict)
                elif isinstance(x, torch.optim.Optimizer):
                    x.load_state_dict(state_dict)
                else:
                    raise TypeError(f"Unsupported type {type(x)}")

        if "optim" in state:
            if self.lr_scheduler is not None and scheduler_state is None:
                self._reconstruct_lr_scheduler_state(
                    checkpoint_epoch=checkpoint_epoch,
                    checkpoint_global_step=checkpoint_global_step,
                )
            self.optim.load_state_dict(state["optim"])

        if self.lr_scheduler is not None:
            if scheduler_state is not None:
                self.lr_scheduler.load_state_dict(scheduler_state)
            self._global_step = (
                int(checkpoint_global_step)
                if checkpoint_global_step is not None
                else self._infer_legacy_global_step(checkpoint_epoch)
            )
        else:
            self._global_step = int(checkpoint_global_step or 0)

        self._checkpoint_epoch = checkpoint_epoch

    def _infer_legacy_global_step(self, checkpoint_epoch):
        completed_epochs = max(int(checkpoint_epoch) + 1, 0)
        if self._scheduler_steps_per_epoch is None:
            return 0
        return completed_epochs * self._scheduler_steps_per_epoch

    def _reconstruct_lr_scheduler_state(self, checkpoint_epoch, checkpoint_global_step=None):
        if self.lr_scheduler is None:
            return

        completed_epochs = max(int(checkpoint_epoch) + 1, 0)
        if self.lr_scheduler_interval == "epoch":
            replay_steps = completed_epochs
            warnings.warn(
                "Checkpoint is missing scheduler state; reconstructing the epoch-based "
                "scheduler from the saved epoch.",
                stacklevel=2,
            )
        else:
            if checkpoint_global_step is not None:
                replay_steps = int(checkpoint_global_step)
                warnings.warn(
                    "Checkpoint is missing scheduler state; reconstructing the step-based "
                    "scheduler from the saved global step.",
                    stacklevel=2,
                )
            elif self._scheduler_steps_per_epoch is not None:
                replay_steps = completed_epochs * self._scheduler_steps_per_epoch
                warnings.warn(
                    "Checkpoint is missing scheduler state; approximating the step-based "
                    "scheduler from epoch * len(train_dl).",
                    stacklevel=2,
                )
            else:
                replay_steps = 0
                warnings.warn(
                    "Checkpoint is missing scheduler state and train_dl is unavailable, so "
                    "the step-based scheduler could not be reconstructed.",
                    stacklevel=2,
                )

        if replay_steps <= 0:
            return

        base_lrs = getattr(self.lr_scheduler, "base_lrs", None)
        if base_lrs is not None:
            for param_group, base_lr in zip(self.optim.param_groups, base_lrs):
                param_group["lr"] = base_lr
                param_group.setdefault("initial_lr", base_lr)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
            )
            for _ in range(replay_steps):
                self.lr_scheduler.step()

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

    def _step_scheduler_on_train_batch_end(self):
        self._global_step += 1
        if self.lr_scheduler is not None and self.lr_scheduler_interval == "step":
            self.lr_scheduler.step()

    def _step_scheduler_on_train_epoch_end(self):
        if self.lr_scheduler is not None and self.lr_scheduler_interval == "epoch":
            self.lr_scheduler.step()

    def run(self):
        print(f"Running {str(self)}")
        epochs: int = self.config["train.epochs"]

        # If using mixed precision, then create a GradScaler to scale gradients during mixed precision training.
        if self.config.get('experiment.torch_mixed_precision', False):
            self.grad_scaler = GradScaler('cuda')


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

            self._step_scheduler_on_train_epoch_end()

            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0:
                self.checkpoint()

            self.run_callbacks("epoch", epoch=epoch)

        self.checkpoint(tag="last")
        self.run_callbacks("wrapup")

    def run_phase(self, phase, epoch):
        dl = getattr(self, f"{phase}_dl")

        grad_enabled = phase == "train"
        self.model.train(grad_enabled)  # For dropout, batchnorm, &c

        # Check if augmentations are enabled
        aug_cfg = None if "augmentations" not in self.config else self.config["augmentations"] 
        augmentation = (phase == "train") and (aug_cfg is not None)

        phase_meters = MeterDict()
        iter_loader = iter(dl)
        
        # Check if profiling is enabled (disabled by default for torch.compile compatibility)
        profile_timing = self.config.get('experiment.profile_timing', False)
        
        # Initialize profiling timers (only used if profile_timing is True)
        profile_times = defaultdict(float) if profile_timing else None
        profile_counts = defaultdict(int) if profile_timing else None

        # Sometimes we want to compute aggregate metrics that work over the entire
        # phase. This requires us to track the predicted y_true and y_pred.
        output_dict = {
            "y_true": [],
            "y_pred": [],
        }
        # And we have a list of quantities that we want to track over epochs.
        log_trackers = self.config.get("log.trackers", None)
        if log_trackers is not None:
            tracker_dict = {
                t_name: [] for t_name in log_trackers
            }
        else:
            tracker_dict = {} 
        # Run the phase.
        with torch.set_grad_enabled(grad_enabled):
            for batch_idx in range(len(dl)):
                # Profile: Data loading (only if profiling enabled)
                if profile_timing:
                    t0 = time.perf_counter()
                batch = next(iter_loader) # Doing this lets us time the data loading.
                if profile_timing:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    profile_times['data_loading'] += time.perf_counter() - t0
                    profile_counts['data_loading'] += 1
                
                # Profile: Run step (forward, backward, optim)
                if profile_timing:
                    t0 = time.perf_counter()
                outputs = self.run_step(
                    batch=batch,
                    backward=grad_enabled,
                    augmentation=augmentation,
                    epoch=epoch,
                    phase=phase,
                    profile_times=profile_times,
                    profile_counts=profile_counts,
                )
                if profile_timing:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    profile_times['total_step'] += time.perf_counter() - t0
                    profile_counts['total_step'] += 1

                # If the outputs is empty, then we skip the metrics computation.
                if len(outputs) > 0:
                    # Profile: Metrics computation (only if profiling enabled)
                    if profile_timing:
                        t0 = time.perf_counter()
                    batch_metrics, batch_metric_weights = self.compute_metrics(outputs)
                    phase_meters.update(
                        batch_metrics, 
                        weights=batch_metric_weights
                    )
                    if profile_timing:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        profile_times['metrics'] += time.perf_counter() - t0
                        profile_counts['metrics'] += 1
                    # Accumulate outputs for global metrics and trackers.
                    if len(self.global_metric_fns) > 0 or len(tracker_dict) > 0:
                        self.accumulate_step_outputs(outputs, output_dict, tracker_dict)
                    # Run the batch-wise callbacks if you have them.
                    self.run_callbacks(
                        "batch", 
                        epoch=epoch, 
                        batch_idx=batch_idx, 
                        phase=phase
                    )
                else:
                    print(f"Warning: No outputs for batch {batch_idx} in {phase} epoch {epoch}.")
        phase_metrics = _materialize_scalar_metrics(
            {"phase": phase, "epoch": epoch, **phase_meters.collect("mean")}
        )

        for t_name in tracker_dict:
            tracker_dict[t_name] = torch.cat(tracker_dict[t_name])

        # Compute the global metrics.
        if len(self.global_metric_fns) > 0:
            # Convert the keys in the output_dict to tensors.
            for key in output_dict:
                output_dict[key] = torch.cat(output_dict[key])
            global_metrics = self.compute_global_metrics(output_dict)
        else:
            global_metrics = {}
        avg_trackers = {t_name: np.round(torch.mean(tracker_dict[t_name]).item(), 4) for t_name in tracker_dict}
        metric_dict = {**phase_metrics, **global_metrics, **avg_trackers}
        self.metrics.log(metric_dict)
        
        # Print profiling results (only if profiling enabled)
        if profile_timing:
            print(f"\n{'='*60}")
            print(f"PROFILING RESULTS - {phase.upper()} Epoch {epoch}")
            print(f"{'='*60}")
            total_time = sum(profile_times.values())
            for key in sorted(profile_times.keys()):
                avg_time = profile_times[key] / max(profile_counts[key], 1) * 1000  # Convert to ms
                total_key_time = profile_times[key]
                percentage = (total_key_time / total_time * 100) if total_time > 0 else 0
                print(f"{key:20s}: {total_key_time:7.3f}s total | {avg_time:7.3f}ms avg | {percentage:5.1f}%")
            print(f"{'='*60}\n")

        return metric_dict

    def run_step(
        self, 
        batch, 
        epoch=None, 
        phase=None,
        backward=True, 
        augmentation=True,
        profile_times=None,
        profile_counts=None,
    ):
        # Profile: Data transfer to device (only if profiling enabled)
        if profile_times is not None:
            t0 = time.perf_counter()
        batch = to_device(
            batch, self.device, self.config.get("train.channels_last", False)
        )
        if profile_times is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile_times['data_to_device'] += time.perf_counter() - t0
            profile_counts['data_to_device'] += 1

        x, y = batch["img"], batch["label"]

        # Profile: Forward pass (only if profiling enabled)
        if profile_times is not None:
            t0 = time.perf_counter()
        y_hat = self.model(x)
        if profile_times is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile_times['forward_pass'] += time.perf_counter() - t0
            profile_counts['forward_pass'] += 1

        # Profile: Loss computation (only if profiling enabled)
        if profile_times is not None:
            t0 = time.perf_counter()
        loss = self.loss_func(y_hat, y)
        if profile_times is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile_times['loss_computation'] += time.perf_counter() - t0
            profile_counts['loss_computation'] += 1

        if backward:
            # Profile: Backward pass (only if profiling enabled)
            if profile_times is not None:
                t0 = time.perf_counter()
            loss.backward()
            if profile_times is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                profile_times['backward_pass'] += time.perf_counter() - t0
                profile_counts['backward_pass'] += 1
            
            # Profile: Optimizer step (only if profiling enabled)
            if profile_times is not None:
                t0 = time.perf_counter()
            self.optim.step()
            self.optim.zero_grad()
            self._step_scheduler_on_train_batch_end()
            if profile_times is not None:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                profile_times['optimizer_step'] += time.perf_counter() - t0
                profile_counts['optimizer_step'] += 1
        
        forward_batch = {
            "loss": loss, "x": x, "y_true": y, "y_pred": y_hat
        }
        # Run step-wise callbacks if you have them.
        self.run_callbacks("step", batch=forward_batch)

        return forward_batch

    def compute_metrics(self, outputs):
        batch_loss = outputs["loss"]
        # If the loss is not a scalar, then we need to reduce it to a scalar.
        if isinstance(batch_loss, Tensor):
            if len(batch_loss.shape) != 0:
                batch_loss = batch_loss.mean()
            batch_loss = batch_loss.item()
        metrics = {"loss": batch_loss}
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
    
    def compute_global_metrics(self, output_dict):
        metrics = {}
        for name, fn in self.global_metric_fns.items():
            y_pred = output_dict["y_pred"]
            y_true = output_dict["y_true"]
            metrics[name] = fn(y_pred, y_true).item()
        return metrics

    def accumulate_step_outputs(self, outputs: dict, output_dict: dict, tracker_dict: dict) -> None:
        """
        Accumulate step outputs into phase-level dictionaries.
        
        Override this method in subclasses to change which keys are accumulated.
        
        Args:
            outputs: Dictionary of outputs from run_step
            output_dict: Dictionary to accumulate y_true and y_pred across the phase
            tracker_dict: Dictionary to accumulate tracker values across the phase
        """
        # Keep track of what our y_true and y_pred are for the entire phase.
        output_dict["y_true"].append(outputs["y_true"])
        output_dict["y_pred"].append(outputs["y_pred"])
        # Then add the trackers to the tracker dictionary.
        for t_name in tracker_dict:
            tracker_dict[t_name].append(outputs[t_name])

    def build_augmentations(self, load_aug_pipeline):
        pass
