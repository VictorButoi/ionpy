# Torch imports
import torch.nn as nn
# Our different augmentation libraries
import importlib
import albumentations as A
import torchvision.transforms as transforms


def absolute_import(reference):
    module, _, attr = reference.rpartition(".")
    if importlib.util.find_spec(module) is not None:
        module = importlib.import_module(module)
        if hasattr(module, attr):
            return getattr(module, attr)

    raise ImportError(f"Could not import {reference}")

def init_torch_transforms(transform_listt):
    transform_list = initialize_transforms(transform_listt)
    if transform_list is None:
        return None
    else:
        return transforms.Compose(transform_list)


def init_album_transforms(transform_list):
    transform_list = initialize_transforms(transform_list)
    if transform_list is None:
        return None
    else:
        return A.Compose(transform_list)


def initialize_transforms(transform_list):
    if transform_list is None:
        return None
    initialize_trsfm_list = []
    for transform in transform_list:
        if isinstance(transform, dict):
            transform_name = list(transform.keys())[0]
            transform_args = transform[transform_name]
            transform_instance = absolute_import(transform_name)(**transform_args)
        else:
            transform_instance = absolute_import(transform)()
        initialize_trsfm_list.append(transform_instance)

    return initialize_trsfm_list