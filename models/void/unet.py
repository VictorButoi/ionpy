from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from pylot.nn import get_nonlinearity
from pylot.nn.hyper import VoidModule, VoidConvBlock


@dataclass(eq=False, repr=False)
class VoidUNet(VoidModule):

    in_channels: int
    out_channels: int
    filters: List[int]
    up_filters: Optional[List[int]] = None
    out_activation: Optional[str] = None
    convs_per_block: int = 1
    skip_connections: bool = True
    batch_norm: bool = True
    dims: int = 2
    interpolation_mode: str = "linear"
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
            batch_norm=self.batch_norm,
            dims=self.dims,
        )
        if self.conv_kws:
            conv_args.update(self.conv_kws)

        for in_ch, out_ch in zip([self.in_channels] + filters[:-1], filters):
            c = VoidConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_args)
            self.down_blocks.append(c)

        prev_out_ch = filters[-1]
        skip_chs = filters[-2::-1]
        for skip_ch, out_ch in zip(skip_chs, up_filters):
            in_ch = skip_ch + prev_out_ch if self.skip_connections else prev_out_ch
            c = VoidConvBlock(in_ch, [out_ch] * self.convs_per_block, **conv_args)
            prev_out_ch = out_ch
            self.up_blocks.append(c)

        # filters[0] == up_filters[-1]
        self.out_conv = VoidConvBlock(
            prev_out_ch,
            [self.out_channels],
            activation=None,
            kernel_size=1,
            dims=self.dims,
            batch_norm=False,
        )

        if self.interpolation_mode == "linear":
            self.interpolation_mode = ["linear", "bilinear", "trilinear"][self.dims - 1]

        if self.out_activation:
            if self.out_activation == "Softmax":
                # For Softmax, we need to specify the channel dimension
                self.out_fn = nn.Softmax(dim=1)
            else:
                self.out_fn = get_nonlinearity(self.out_activation)()

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
            if self.skip_connections:
                x = torch.cat([x, conv_outputs[-i]], dim=1)
            x = conv_block(x)

        x = self.out_conv(x)
        if self.out_activation:
            x = self.out_fn(x)

        return x
