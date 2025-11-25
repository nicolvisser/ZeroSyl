# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Normalization modules."""

import typing as tp

import torch
from torch import nn


class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """

    def __init__(
        self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs
    ):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        # Move channels to last dimension
        x = x.permute(0, -1, *range(1, x.dim() - 1))
        x = super().forward(x)
        # Move channels back to original position
        x = x.permute(0, *range(2, x.dim()), 1)
        return x
