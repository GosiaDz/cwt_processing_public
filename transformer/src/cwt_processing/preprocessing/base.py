"""Base class and interface for spectrum signal preprocessing blocks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import torch
from torch import nn


class BasePreprocessing(nn.Module, ABC):
    """Abstract scaffold for signal preprocessing.

    A preprocessing block transforms raw 1D signals into feature maps for a neural network,
    returning both the feature tensor and a context dictionary (e.g. normalization statistics).
    """

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of feature channels generated per spectral point."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process raw 1-D signal into features and context.

        Args:
            x: Raw input tensor of shape `(batch_size, num_points)`.

        Returns:
            features: Feature tensor, typically `(batch_size, channels, length)`.
            context: Dictionary containing normalization parameters or metadata (e.g. `xmin`, `xmax`).
        """
        raise NotImplementedError
