# Local imports
from ..experiment.util import absolute_import
from .containers import augmentations_from_config
# Torch imports
import torch.nn as nn
# Our different augmentation libraries
import albumentations as A
import torchvision.transforms as transforms


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


def init_kornia_transforms(transform_list, mode=None):
    if transform_list is None:
        return None
    else:
        if mode == 'image':
            transform_list = initialize_transforms(transform_list)
            return nn.Sequential(*transform_list)  # Kornia requires a module-based composition
        else:
            return augmentations_from_config(transform_list)


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