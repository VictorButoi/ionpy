# Augmentation library imports
import albumentations as A
import monai.transforms as mt
import torchvision.transforms as T
# Mics imports
import inspect
import torch.nn as nn
# Local imports
from .transform_wrappers import initialize_transforms
from .containers import augmentations_from_config
from .voxynth import build_voxynth_aug_pipeline

class CombinedPipeline:
    """
    Sequentially applies a list of pipelines (skipping any that are None).
    """
    def __init__(self, pipelines):
        self.pipelines = [p for p in pipelines if p is not None]

    def __call__(self, data):
        for pipe in self.pipelines:
            data = pipe(data)
        return data

class SequentialIgnoreSecond(nn.Sequential):
    def forward(self, x, y):  # Ignore the second argument
        return super().forward(x), y


def build_monai_pipeline(transform_list):
    """
    Given a list like:
      [{"RandFlipd": {...}},
       {"RandAdjustContrastd": {"keys": [...], "prob": 0.15, "gamma_range": [0.75,1.25]}},
       ...]
    this will:
      - auto-filter unsupported kwargs (gamma_range in this example)
      - or raise a clear TypeError explaining what went wrong
    """
    if not transform_list:
        return None

    ops = []
    for t in transform_list:
        name, params = next(iter(t.items()))
        cls = getattr(mt, name)
        sig = inspect.signature(cls.__init__)
        # drop 'self' and any unexpected keys
        valid_keys = set(sig.parameters) - {"self", "kwargs"}
        filtered = {k: v for k, v in params.items() if k in valid_keys}
        bad_keys = set(params) - valid_keys

        if bad_keys:
            raise TypeError(
                f"{name} got unexpected argument(s): {bad_keys!r}\n"
                f"Valid args are: {sorted(valid_keys)}"
            )
        ops.append(cls(**filtered))
    return mt.Compose(ops)

def build_albumentations_pipeline(transform_list):
    if not transform_list:
        return None
    ops = []
    for t in transform_list:
        name, params = next(iter(t.items()))
        cls = getattr(A, name)
        ops.append(cls(**params))
    return A.Compose(ops)

# extend this dict if you want to support more libs
TRANSFORM_BUILDERS = {
    "voxynth":     lambda cfg, **kw: build_voxynth_aug_pipeline(cfg),
    "kornia":      lambda cfg, **kw: init_kornia_transforms(cfg, **kw),
    "monai":       lambda cfg, **kw: build_monai_pipeline(cfg),
    "albumentations": lambda cfg, **kw: build_albumentations_pipeline(cfg),
    "torchvision": lambda cfg, **kw: T.Compose([getattr(T, name)(**params)
                                               for t in cfg
                                               for name, params in t.items()]),
}

def init_kornia_transforms(transform_list, mode=None):
    if transform_list is None:
        return None
    else:
        if mode == 'image':
            transform_list = initialize_transforms(transform_list)
            return SequentialIgnoreSecond(*transform_list)  # Kornia requires a module-based composition
        else:
            return augmentations_from_config(transform_list)

def build_gpu_aug_pipeline(gpu_aug_cfg):
    if not gpu_aug_cfg:
        return None
    pipelines = []
    for lib, builder in TRANSFORM_BUILDERS.items():
        lib_cfg = gpu_aug_cfg.get(lib)
        if not lib_cfg:
            continue
        # pass extra args (e.g. mode for Kornia)
        extra_args = {}
        if lib == "kornia":
            extra_args["mode"] = gpu_aug_cfg.get("mode", "both")

        pipelines.append(builder(lib_cfg, **extra_args))

    return CombinedPipeline(pipelines)
