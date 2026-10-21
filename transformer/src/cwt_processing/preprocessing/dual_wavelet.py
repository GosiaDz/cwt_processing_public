"""Dual-wavelet preprocessing block implementing Mexican Hat and Gaussian-Dedicated kernels."""
from __future__ import annotations

import math
from typing import Any, Sequence
import torch
import torch.nn.functional as F
from torch import nn

from cwt_processing.preprocessing.base import BasePreprocessing


def _gaussian_dedicated_kernel(scale: float) -> torch.Tensor:
    """Evaluate Gaussian-Dedicated (GD) wavelet kernel at given scale."""
    radius = int(math.ceil(4 * scale))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    values = torch.exp(-4.0 * math.log(2.0) * (t / float(scale)) ** 2)
    values = values - values.mean()
    return values / (torch.linalg.vector_norm(values) + 1e-12)


def _mexican_hat_kernel(scale: float) -> torch.Tensor:
    """Evaluate Mexican Hat (MH) wavelet kernel at given scale."""
    radius = int(math.ceil(4 * scale))
    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    u = t / float(scale)
    values = (1.0 - u * u) * torch.exp(-0.5 * u * u)
    values = values - values.mean()
    return values / (torch.linalg.vector_norm(values) + 1e-12)


class DualWaveletPreprocessing(BasePreprocessing):
    """Dual-wavelet preprocessing module computing MH and GD channels at specified scales.

    Default scales are (5, 23), producing 4 channels: [MH5, MH23, GD5, GD23].
    Boundary handling replicates edge samples to eliminate boundary artifacts.
    """

    def __init__(
        self,
        scales: Sequence[float] = (5.0, 23.0),
        normalize_input: bool = False,
    ) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.normalize_input = normalize_input

        # Register MH and GD kernels as buffers
        self.kernels: list[tuple[str, float, torch.Tensor]] = []
        idx = 0
        for s in self.scales:
            k_mh = _mexican_hat_kernel(s).view(1, 1, -1)
            self.register_buffer(f"kernel_{idx}", k_mh)
            idx += 1

        for s in self.scales:
            k_gd = _gaussian_dedicated_kernel(s).view(1, 1, -1)
            self.register_buffer(f"kernel_{idx}", k_gd)
            idx += 1

        self._num_channels = idx

    @property
    def out_channels(self) -> int:
        """Total number of wavelet channels (len(scales) * 2)."""
        return self._num_channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        xmin = x.amin(dim=-1, keepdim=True)
        xmax = x.amax(dim=-1, keepdim=True)

        if self.normalize_input:
            denom = torch.clamp(xmax - xmin, min=1e-8)
            x_in = (x - xmin) / denom
        else:
            x_in = x

        x_sig = x_in.unsqueeze(1).to(dtype=torch.float32)

        channels: list[torch.Tensor] = []
        for idx in range(self._num_channels):
            kernel: torch.Tensor = getattr(self, f"kernel_{idx}")
            radius = (kernel.shape[-1] - 1) // 2
            padded = F.pad(x_sig, (radius, radius), mode="replicate")
            channels.append(F.conv1d(padded, kernel))

        features = torch.cat(channels, dim=1).to(dtype=x.dtype)
        context: dict[str, Any] = {
            "xmin": xmin,
            "xmax": xmax,
            "scales": self.scales,
            "out_channels": self.out_channels,
        }
        return features, context
