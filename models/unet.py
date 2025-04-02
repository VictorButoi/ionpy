# Torch imports 
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
# Local imports
from ..nn import get_nonlinearity, ConvBlock
# Misc imports
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union


@dataclass(eq=False, repr=False)
class UNet(nn.Module):

    in_channels: int
    out_channels: int
    filters: List[int]
    dims: int = 2
    convs_per_block: int = 1
    skip_connections: Union[bool, List[bool]] = True
    interpolation_mode: str = "linear"
    out_activation: Optional[str] = None
    up_filters: Optional[List[int]] = None
    conv_kws: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__init__()

        filters = list(self.filters)
        if self.up_filters is None:
            self.up_filters = filters[-2::-1]
        assert len(self.up_filters) == len(self.filters) - 1
        up_filters = list(self.up_filters)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        conv_args = dict(
            dims=self.dims,
        )
        if self.conv_kws:
            conv_args.update(self.conv_kws)

        for in_ch, out_ch in zip([self.in_channels] + filters[:-1], filters):
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_args)
            self.down_blocks.append(c)

        # If our skip connections are just a boolean, we need to create a list of the same length as the number of levels*
        if isinstance(self.skip_connections, bool):
            self.skip_connections = [self.skip_connections] * len(up_filters)
        assert len(self.skip_connections) == len(up_filters),\
            f"Skip connections list must be the same length as the number of levels. Got {len(self.skip_connections)} and {len(up_filters)}"

        prev_out_ch = filters[-1]
        skip_chs = filters[-2::-1]
        for l_idx, (skip_ch, out_ch) in enumerate(zip(skip_chs, up_filters)):
            in_ch = skip_ch + prev_out_ch if self.skip_connections[-(l_idx + 1)] else prev_out_ch # We go from the end because the first index is the top level.
            c = ConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_args)
            prev_out_ch = out_ch
            self.up_blocks.append(c)

        # filters[0] == up_filters[-1]
        self.out_conv = ConvBlock(
            prev_out_ch,
            [self.out_channels],
            activation=None,
            kernel_size=1,
            dims=self.dims,
            norm=None,
        )

        if self.interpolation_mode == "linear":
            self.interpolation_mode = ["linear", "bilinear", "trilinear"][self.dims - 1]

        if self.out_activation:
            if self.out_activation == "Softmax":
                # For Softmax, we need to specify the channel dimension
                self.out_fn = nn.Softmax(dim=1)
            else:
                self.out_fn = get_nonlinearity(self.out_activation)()

        self.reset_parameters()

    def reset_parameters(self):
        for group in (self.down_blocks, self.up_blocks, [self.out_conv]):
            for module in group:
                module.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:

        conv_outputs = []
        for i, conv_block in enumerate(self.down_blocks):
            x = conv_block(x)
            if i == len(self.down_blocks) - 1:
                break
            conv_outputs.append(x)
            x = F.max_pool2d(x, 2)

        for i, conv_block in enumerate(self.up_blocks, start=1):
            x = F.interpolate(
                x,
                size=conv_outputs[-i].size()[-self.dims :],
                align_corners=True,
                mode=self.interpolation_mode,
            )
            if self.skip_connections[-i]: # We go from the end because the first index is the top level.
                x = torch.cat([x, conv_outputs[-i]], dim=1)
            x = conv_block(x)

        x = self.out_conv(x)
        if self.out_activation:
            x = self.out_fn(x)

        return x
