"""Abstract base class for spectrum neural networks wrapped in PyTorch Lightning."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from cwt_processing.preprocessing.base import BasePreprocessing


class BaseSpectrumNetwork(L.LightningModule, ABC):
    """Base Lightning module managing preprocessing, backbone forward, and training loop.

    Each network in its structure uses one of the backbones from backbones.py.
    """

    preprocessing: BasePreprocessing
    backbone: nn.Module

    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        features, context = self.preprocessing(x)
        predictions = self.backbone(features)
        return predictions, context

    @abstractmethod
    def process_targets(self, raw_targets: torch.Tensor, context: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode_predictions(
        self,
        predictions: torch.Tensor,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        return predictions

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.trainer is not None:
            train_dataloader = getattr(self.trainer, "train_dataloader", None)
            if train_dataloader is not None:
                ds = getattr(train_dataloader, "dataset", None)
                if ds is not None and hasattr(ds, "set_epoch"):
                    ds.set_epoch(self.current_epoch)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        signals, raw_targets = batch
        predictions, context = self(signals)
        targets = self.process_targets(raw_targets, context)
        loss = self.loss_function(predictions, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        signals, raw_targets = batch
        predictions, context = self(signals)
        targets = self.process_targets(raw_targets, context)
        loss = self.loss_function(predictions, targets)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        signals, raw_targets = batch
        predictions, context = self(signals)
        targets = self.process_targets(raw_targets, context)
        loss = self.loss_function(predictions, targets)
        self.log("test_loss", loss)
        self.log("test_mae", F.l1_loss(predictions, targets))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
