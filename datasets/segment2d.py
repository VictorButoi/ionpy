# misc imports
import parse
import torch
import einops
import warnings
import numpy as np
from parse import parse
from dataclasses import dataclass
from pydantic import validate_arguments
from typing import Any, List, Literal, Optional
# Local imports
from .path import DatapathMixin
from .thunder import ThunderDataset
from ..util.thunder import UniqueThunderReader
from ..augmentation import init_album_transforms
from ..util.validation import validate_arguments_init


def parse_task(task):
    return parse("{dataset}/{group}/{modality}/{axis}", task).named


@validate_arguments_init
@dataclass
class Segment2D(ThunderDataset, DatapathMixin):
    # task is (dataset, group, modality, axis)
    # - optionally label but see separate arg
    task: str
    resolution: Literal[64, 128, 256]
    split: Literal["train", "val", "test"] = "train"
    label: Optional[int] = None
    mode: Literal["bw", "rgb"] = "bw"
    label_type: Literal["instance", "soft", "hard", "multiannotator"] = "soft"
    slicing: Literal["midslice", "maxslice"] = "midslice"
    version: str = "v4.2"
    min_label_density: float = 0.0
    background: bool = False
    transforms: Optional[Any] = None
    num_examples: Optional[int] = None
    iters_per_epoch: Optional[int] = None
    preload: bool = False
    data_root: Optional[str] = None
    return_data_id: bool = False

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()

        # Data Validation
        msg = "Background is only supported for multi-label"
        assert not (self.label is not None and self.background), msg

        if self.slicing == "maxslice" and self.label is None:
            raise ValueError("Must provide label, when segmenting maxslices")

        # min_label_density
        subjects: List[str] = self._db["_splits"][self.split]
        if self.min_label_density > 0.0:
            label_densities: np.ndarray = self._db["_label_densities"]
            if self.label is None:
                label_density = label_densities.mean(axis=1)
            else:
                label_density = label_densities[:, self.label]
            all_subjects = np.array(self._db["_subjects"])
            valid_subjects = set(all_subjects[label_density > self.min_label_density])
            subjects = [s for s in subjects if s in valid_subjects]

        # If num_examples is not None, then subset the dataset
        if self.num_examples is not None:
            if self.num_examples > len(subjects):
                raise ValueError(
                    f"num_examples {self.num_examples} is greater than the number of samples {len(subjects)}"
                )
            subjects = subjects[: self.num_examples]

        print("Number of subjects: ", len(subjects))
        self.samples = subjects
        self.subjects = subjects

        # Signature to file checking
        file_attrs = self.attrs
        for key, val in parse_task(init_attrs["task"]).items():
            if file_attrs[key] != val:
                raise ValueError(
                    f"Attr {key} mismatch init:{val}, file:{file_attrs[key]}"
                )
        for key in ("resolution", "slicing", "version"):
            if init_attrs[key] != file_attrs[key]:
                raise ValueError(
                    f"Attr {key} mismatch init:{init_attrs[key]}, file:{file_attrs[key]}"
                )
        
        # Build the augmentation pipeline
        self.transforms_pipeline = init_album_transforms(self.transforms)

    def __len__(self):
        if self.iters_per_epoch:
            return self.iters_per_epoch
        return len(self.samples)

    def __getitem__(self, key):
        if self.iters_per_epoch:
            key %= len(self.samples)

        img, seg = super().__getitem__(key)
        assert img.dtype == np.float32
        if self.label_type in ['soft','multiannotator']:
            assert seg.dtype == np.float32
        else:
            assert seg.dtype == np.int8, print(seg.dtype)

        if self.slicing == "maxslice":
            img = img[self.label]
        img = img[None]
        if self.label is not None:
            if self.label_type in ['soft','multiannotator']:
                seg = seg[self.label : self.label + 1]
            elif self.label_type == 'hard':
                seg = (seg == self.label).astype(np.float32)
            elif self.label_type == 'instance':
                # For universeg, take all instances
                seg = seg[self.label : self.label + 1]
                seg = (seg > 0).astype(np.float32)
        # Squeeze the image and mask. This is a relic of the fact that we did
        # our processing with [1, H, W].
        img = img.squeeze()
        seg = seg.squeeze()
        # Get the class name
        if self.transforms:
            transform_obj = self.transforms_pipeline(
                image=img,
                mask=seg
            )
            img, seg = transform_obj["image"], transform_obj["mask"]
        else:
            # We need to convert these image and masks to tensors at least.
            img = torch.tensor(img)
            seg = torch.tensor(seg)

        # If our mode is rgb, then we need to stack our image 3X on the channel dim
        if self.mode == "rgb":
            img = np.concatenate([img] * 3)
                    
        if self.background:
            if self.label_type in ['soft','multiannotator']:
                bg = 1 - seg.sum(axis=0, keepdims=True)
                seg = np.concatenate([bg, seg])
            else:
                raise NotImplementedError
        elif self.label is not None:
            seg = seg[None] # Add the channel dim
        
        # Prepare return dictionary
        return_dict = {
            "img": img,
            "label": seg 
        }
        # Add the data_id if necessary
        if self.return_data_id:
            return_dict["data_id"] = self.subjects[key] 
        return return_dict

    @property
    def num_labels(self) -> int:
        if self.label is not None:
            return 1
        return self.attrs["n_labels"] + self.background

    @property
    def _folder_name(self):
        return f"megamedical/{self.version}/res{self.resolution}/{self.slicing}/{self.task}"

    @classmethod
    def frompath(cls, path, **kwargs):
        _, relpath = str(path).split("megamedical/")

        kwargs.update(
            parse("{version}/res{resolution:d}/{slicing:w}/{task}", relpath).named
        )
        return cls(**kwargs)

    @classmethod
    def fromfile(cls, path, **kwargs):
        a = UniqueThunderReader(path)["_attrs"]
        task = f"{a['dataset']}/{a['group']}/{a['modality']}/{a['axis']}"
        return cls(
            task=task,
            resolution=a["resolution"],
            slicing=a["slicing"],
            version=a["version"],
            **kwargs,
        )

    def other_split(self, split):
        if split == self.split:
            return self
        return Segment2D(
            split=split,
            # everything is the same bar the split
            task=self.task,
            resolution=self.resolution,
            label=self.label,
            slicing=self.slicing,
            version=self.version,
            min_label_density=self.min_label_density,
            background=self.background,
            preload=self.preload,
            iters_per_epoch=self.iters_per_epoch,
        )

    @property
    def signature(self):
        return {
            "task": self.task,
            "resolution": self.resolution,
            "split": self.split,
            "label": self.label,
            "slicing": self.slicing,
            "version": self.version,
            "min_label_density": self.min_label_density,
            **parse_task(self.task),
        }