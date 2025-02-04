# Torch imports
from torch import nn
# Misc imports
import einops as E
import numpy as np
from typing import Any, Dict, List, Literal, Optional, Union
# Local imports
from . import paired


class PairedSequential(nn.Sequential):
    """Given a list of augmentation modules with segmentation API
    f(x, y) -> x, y
    it applies them one after the other depending on the value of random_apply
    - False -> applies them sequentially
    - True -> applies all of them in a random order
    - n: int -> applies a random subset of N augmentations
    - (n: int, b: int) -> applies a random subset of randint(n,m)
    """

    def __init__(
        self,
        *augmentations: list[nn.Module],
        random_apply: Union[int, bool, tuple[int, int]] = False,
    ):
        super().__init__()

        self.random_apply = random_apply

        for i, augmentation in enumerate(augmentations):
            self.add_module(f"{augmentation.__class__.__name__}_{i}", augmentation)

    def _get_idxs(self):

        N = len(self)

        if self.random_apply is False:
            return np.arange(N)
        elif self.random_apply is True:
            return np.random.permutation(N)
        elif isinstance(self.random_apply, int):
            assert 1 <= self.random_apply <= len(self)
            return np.random.permutation(N)[: self.random_apply]
        elif isinstance(self.random_apply, tuple):
            n = np.random.randint(*self.random_apply)
            return np.random.permutation(N)[:n]
        else:
            raise TypeError(f"Invalid type {type(self.random_apply)}")

    def forward(self, image, label):
        for i in self._get_idxs():
            image, label = self[i](image, label)
        return image, label 


class VerbosePairedSequential(PairedSequential):
    @classmethod
    def from_existing(cls, segseq):
        return cls(*list(segseq), random_apply=segseq.random_apply)

    def forward(self, image, segmentation):
        samples = []
        for i in self._get_idxs():
            image, segmentation = self[i](image, segmentation)
            samples.append([image, segmentation, self[i]._params])
        return samples


def augmentations_from_config(config: List[Dict[str, Any]]) -> PairedSequential:

    augmentations = []

    random_apply = False
    for aug in config:
        if not isinstance(aug, dict):
            if "Random" in aug:
                aug = {aug: {"p": 1.0}}
            else:
                raise ValueError(f"Invalid augmentation {aug}")
        else:
            assert len(aug) == 1 and isinstance(aug, dict)
        for full_name, params in aug.items():
            name = full_name.split(".")[-1]
            if name == "random_apply":
                random_apply = params
            else:
                augmentations.append(getattr(paired, name)(**params))

    return PairedSequential(*augmentations, random_apply=random_apply)
