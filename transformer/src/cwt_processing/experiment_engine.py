
"""Standalone synthetic-peak experiment engine.

Drop this file and synthetic_peak_generator.py into:
    src/cwt_processing/

It reuses the existing project:
    - PatchTransformerBackbone
    - BaseSpectrumNetwork
    - run_generalized_reporting() -> report.html + metrics.json

It does NOT use per-spectrum min-max normalization.

Examples
--------
Default publication-like baseline:
    python -m cwt_processing.experiment_engine

Dual MH+GD, scales 5 and 23:
    python -m cwt_processing.experiment_engine --wavelets "MH;GD" --scales "5;23"

Scale syntax:
    --scales "5;23;46"
    --scales "(21-30;2)"
    --scales "(20-30;5);(55-100;1)"
    --scales "(20;30;5);46"

Different scales per wavelet:
    --wavelets "MH;GD;morl" \
    --wavelet-scales "MH=5;23|GD=(1-9;2)|morl=(10-30;5)"

Pseudo-Voigt (8 targets automatically):
    python -m cwt_processing.experiment_engine --family pseudovoigt

Harder wavelets
---------------
The engine accepts PTWT/PyWavelets continuous-wavelet names, including:
    mexh, morl, gaus1..gaus8, cgau1..cgau8,
    cmorB-C, shanB-C, fbspM-B-C

Examples:
    cmor1.5-1.0
    shan1.5-1.0
    fbsp2-1.0-0.5

Complex CWTs (cmor/shan/fbsp/cgau) are converted to magnitude by default.
Use --complex-mode real_imag to preserve real and imaginary parts as two channels
per scale.  MH and GD are the project's custom real kernels.

Important
---------
Input normalization is deliberately absent. Target standardization is computed
ONCE from the training set and then frozen for validation/test/reporting.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from cwt_processing.networks.backbones import PatchTransformerBackbone
from cwt_processing.networks.base import BaseSpectrumNetwork
from cwt_processing.preprocessing.base import BasePreprocessing
from cwt_processing.preprocessing.dual_wavelet import (
    _gaussian_dedicated_kernel,
    _mexican_hat_kernel,
)
from cwt_processing.synthetic_peak_generator import (
    build_synthetic_datasets,
    flatten_targets,
    output_names,
    outputs_for_family,
)
from cwt_processing.verification import run_generalized_reporting


DEFAULT_TRAIN_SEED = 31337
DEFAULT_VAL_SEED = 7331
DEFAULT_TEST_SEED = 1337

RECOMMENDED_WAVELETS = ("MH", "GD", "mexh", "morl", "gaus1", "gaus2")
KNOWN_STANDARD_WAVELETS = tuple(
    ["mexh", "morl"]
    + [f"gaus{i}" for i in range(1, 9)]
    + [f"cgau{i}" for i in range(1, 9)]
)
COMPLEX_PREFIXES = ("cmor", "shan", "fbsp", "cgau")


def _split_top_level(text: str, separator: str = ";") -> list[str]:
    """Split while ignoring separators inside parentheses."""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in {text!r}.")
        if ch == separator and depth == 0:
            token = "".join(buf).strip()
            if token:
                parts.append(token)
            buf = []
        else:
            buf.append(ch)
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in {text!r}.")
    token = "".join(buf).strip()
    if token:
        parts.append(token)
    return parts


def _inclusive_range(start: float, end: float, step: float) -> list[float]:
    if not all(math.isfinite(v) for v in (start, end, step)):
        raise ValueError("Scale values must be finite.")
    if start <= 0 or end <= 0 or step <= 0 or end < start:
        raise ValueError("Require positive scales/step and end >= start.")
    values: list[float] = []
    x = start
    tol = max(1e-12, abs(end) * 1e-12)
    while x <= end + tol:
        values.append(float(x))
        x += step
        if len(values) > 100_000:
            raise ValueError("Scale range is too large.")
    return values


def parse_scales(spec: str) -> tuple[float, ...]:
    """Parse explicit scales and ranges.

    Supported:
      5;23;46
      (21-30;2)       -> 21,23,25,27,29
      (20;30;5)       -> 20,25,30
      (20-30;5);46
      (20-30;5);(55-100;1)

    Duplicates are removed while preserving order.
    """
    if not spec or not spec.strip():
        raise ValueError("Scale specification cannot be empty.")

    values: list[float] = []
    for token in _split_top_level(spec.strip(), ";"):
        token = token.strip()
        if token.startswith("(") and token.endswith(")"):
            body = token[1:-1].strip()
            # Form: start-end;step
            m = re.fullmatch(
                r"\s*([0-9]*\.?[0-9]+)\s*-\s*([0-9]*\.?[0-9]+)\s*;\s*([0-9]*\.?[0-9]+)\s*",
                body,
            )
            if m:
                start, end, step = map(float, m.groups())
                values.extend(_inclusive_range(start, end, step))
                continue
            # Form: start;end;step
            fields = [v.strip() for v in body.split(";")]
            if len(fields) == 3:
                start, end, step = map(float, fields)
                values.extend(_inclusive_range(start, end, step))
                continue
            raise ValueError(
                f"Invalid scale range {token!r}. Use '(start-end;step)' "
                "or '(start;end;step)'."
            )
        else:
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(f"Invalid scale token: {token!r}") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Scales must be finite and > 0.")
            values.append(value)

    deduped: list[float] = []
    for v in values:
        if not any(math.isclose(v, old, rel_tol=1e-12, abs_tol=1e-12) for old in deduped):
            deduped.append(v)
    if not deduped:
        raise ValueError("No scales resolved.")
    return tuple(deduped)


def parse_wavelets(spec: str) -> tuple[str, ...]:
    values = tuple(v.strip() for v in spec.split(";") if v.strip())
    if not values:
        raise ValueError("At least one wavelet is required.")
    return values


def parse_wavelet_scales(
    wavelets: Sequence[str],
    global_scales: str,
    per_wavelet: str | None,
) -> dict[str, tuple[float, ...]]:
    """Resolve global or per-wavelet scale specifications.

    Per-wavelet syntax uses "|" between assignments because ";" belongs to the
    scale grammar, e.g.:
        MH=5;23|GD=(1-9;2)|morl=(10-30;5)
    """
    resolved = {w: parse_scales(global_scales) for w in wavelets}
    if not per_wavelet:
        return resolved

    assignments = [x.strip() for x in per_wavelet.split("|") if x.strip()]
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"Invalid --wavelet-scales item {assignment!r}. "
                "Expected NAME=SCALE_SPEC."
            )
        name, scale_spec = assignment.split("=", 1)
        name = name.strip()
        if name not in wavelets:
            raise ValueError(
                f"Wavelet {name!r} has a scale override but is not listed in --wavelets."
            )
        resolved[name] = parse_scales(scale_spec.strip())
    return resolved


def parse_split(spec: str, total: int) -> tuple[int, int, int]:
    """Parse train/val/test fractions, e.g. 0.8;0.1;0.1."""
    parts = [float(v.strip()) for v in spec.split(";") if v.strip()]
    if len(parts) != 3:
        raise ValueError("--split must contain three fractions: train;val;test.")
    if any(v <= 0 for v in parts):
        raise ValueError("Split fractions must be positive.")
    s = sum(parts)
    fractions = [v / s for v in parts]
    n_train = int(round(total * fractions[0]))
    n_val = int(round(total * fractions[1]))
    n_test = total - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split produced an empty dataset.")
    return n_train, n_val, n_test


def _is_custom_wavelet(name: str) -> bool:
    return name.strip().upper() in {"MH", "GD"}


def _is_complex_wavelet(name: str) -> bool:
    low = name.strip().lower()
    return low.startswith(COMPLEX_PREFIXES)


class MultiWaveletPreprocessing(BasePreprocessing):
    """Concatenate arbitrary wavelet/scale channels without input normalization."""

    def __init__(
        self,
        wavelet_scales: dict[str, Sequence[float]],
        *,
        complex_mode: str = "magnitude",
    ) -> None:
        super().__init__()
        if complex_mode not in {"magnitude", "real_imag"}:
            raise ValueError("complex_mode must be 'magnitude' or 'real_imag'.")
        if not wavelet_scales:
            raise ValueError("wavelet_scales cannot be empty.")

        self.wavelet_scales = {
            str(name): tuple(float(v) for v in scales)
            for name, scales in wavelet_scales.items()
        }
        self.complex_mode = complex_mode

        self._custom_kernels: list[tuple[str, float, str]] = []
        idx = 0
        for wavelet, scales in self.wavelet_scales.items():
            key = wavelet.upper()
            if key in {"MH", "GD"}:
                for scale in scales:
                    kernel = (
                        _mexican_hat_kernel(scale)
                        if key == "MH"
                        else _gaussian_dedicated_kernel(scale)
                    ).view(1, 1, -1)
                    attr = f"custom_kernel_{idx}"
                    self.register_buffer(attr, kernel)
                    self._custom_kernels.append((wavelet, scale, attr))
                    idx += 1

        out = 0
        for wavelet, scales in self.wavelet_scales.items():
            multiplier = 2 if (_is_complex_wavelet(wavelet) and complex_mode == "real_imag") else 1
            out += len(scales) * multiplier
        self._out_channels = out

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def num_channels(self) -> int:
        return self._out_channels

    def _prepare_signal(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 2:
            raise ValueError(
                "Expected signal shape (length,), (batch,length), or (batch,1,length)."
            )
        if not x.is_floating_point():
            x = x.float()
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        x = self._prepare_signal(x)
        x_sig = x.unsqueeze(1).to(dtype=torch.float32)
        channels: list[torch.Tensor] = []

        # Custom project MH/GD kernels.
        custom_lookup = {(w, s): attr for w, s, attr in self._custom_kernels}
        for wavelet, scales in self.wavelet_scales.items():
            if _is_custom_wavelet(wavelet):
                for scale in scales:
                    attr = custom_lookup[(wavelet, scale)]
                    kernel: torch.Tensor = getattr(self, attr)
                    radius = (kernel.shape[-1] - 1) // 2
                    padded = F.pad(x_sig, (radius, radius), mode="replicate")
                    channels.append(F.conv1d(padded, kernel).to(dtype=x.dtype))
                continue

            # Standard PTWT/PyWavelets CWT. Imported lazily so MH/GD-only
            # experiments do not require PTWT at runtime.
            try:
                import ptwt
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Standard CWT wavelets require the project's 'ptwt' dependency. "
                    "MH and GD do not."
                ) from exc
            scale_tensor = torch.tensor(scales, dtype=torch.float32, device="cpu")
            coeffs, _ = ptwt.cwt(x, scale_tensor, wavelet)
            # PTWT returns (scales, batch, length).
            coeffs = coeffs.permute(1, 0, 2).to(device=x.device)

            if torch.is_complex(coeffs):
                if self.complex_mode == "magnitude":
                    coeffs = torch.abs(coeffs)
                else:
                    channels.append(coeffs.real.to(dtype=x.dtype))
                    channels.append(coeffs.imag.to(dtype=x.dtype))
                    continue
            channels.append(coeffs.to(dtype=x.dtype))

        if not channels:
            raise RuntimeError("No preprocessing channels were produced.")
        features = torch.cat(channels, dim=1)

        context = {
            "wavelets": list(self.wavelet_scales.keys()),
            "wavelet_scales": {k: list(v) for k, v in self.wavelet_scales.items()},
            "normalize_input": False,
            "complex_mode": self.complex_mode,
            "out_channels": self.out_channels,
        }
        return features, context


class EnginePeakNetwork(BaseSpectrumNetwork):
    """Small patch Transformer with frozen train-set target standardization."""

    def __init__(
        self,
        *,
        preprocessing: BasePreprocessing,
        signal_length: int,
        n_outputs: int,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
        patch_size: int = 16,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = "plateau",
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.5,
        min_lr: float = 1e-6,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay)
        if n_outputs not in {6, 8}:
            raise ValueError("n_outputs must be 6 or 8.")
        self.preprocessing = preprocessing
        self.backbone = PatchTransformerBackbone(
            in_channels=preprocessing.out_channels,
            out_features=n_outputs,
            patch_size=patch_size,
            signal_length=signal_length,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.n_outputs = n_outputs
        self.output_names = output_names(n_outputs)
        self.scheduler_name = scheduler
        self.scheduler_patience = int(scheduler_patience)
        self.scheduler_factor = float(scheduler_factor)
        self.min_lr = float(min_lr)

        self.register_buffer("target_mean", target_mean.detach().float().clone())
        self.register_buffer("target_std", target_std.detach().float().clone().clamp_min(1e-8))

    def process_targets(
        self, raw_targets: torch.Tensor, context: dict[str, Any]
    ) -> torch.Tensor:
        flat = flatten_targets(raw_targets, self.n_outputs).to(self.target_mean)
        return (flat - self.target_mean) / self.target_std

    def decode_predictions(
        self, predictions: torch.Tensor, context: dict[str, Any] | None = None
    ) -> torch.Tensor:
        return predictions * self.target_std + self.target_mean

    def loss_function(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predictions, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        name = self.scheduler_name.lower()
        if name in {"none", "off"}:
            return optimizer
        if name in {"plateau", "reduce_on_plateau", "reducelronplateau"}:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, int(self.trainer.max_epochs)), eta_min=self.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        raise ValueError(f"Unsupported scheduler: {self.scheduler_name!r}")


def _compute_target_stats(dataset, n_outputs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact train-set target mean/std once, without generating signals."""
    rows = [flatten_targets(dataset.targets_only(i).unsqueeze(0), n_outputs)[0] for i in range(len(dataset))]
    matrix = torch.stack(rows, dim=0).float()
    return matrix.mean(dim=0), matrix.std(dim=0).clamp_min(1e-8)


def _unique_run_dir(base: str | Path) -> Path:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    candidate = base / stamp
    i = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{i}"
        i += 1
    candidate.mkdir(parents=True)
    return candidate


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def run_experiment(
    *,
    family: str = "gaussian",
    samples: int = 50_000,
    split: str = "0.8;0.1;0.1",
    n_points: int = 400,
    x_min: float = 0.0,
    x_max: float = 400.0,
    center_mode: str = "fixed",
    delta_mu_min: float = 10.0,
    delta_mu_max: float | None = None,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
    sigma_min_frac: float = 0.02,
    sigma_max_frac: float = 0.06,
    sigma_ratio_min: float = 0.5,
    sigma_ratio_max: float = 2.0,
    delta_width_min: float = 0.25,
    delta_width_max: float = 3.0,
    amplitude_high_min: float = 0.5,
    amplitude_high_max: float = 1.0,
    amplitude_ratio_min: float = 0.1,
    amplitude_ratio_max: float = 1.0,
    eta_min: float = 0.0,
    eta_max: float = 1.0,
    edge_sigma_margin: float = 4.0,
    noise_level: float = 0.0,
    wavelets: str = "MH",
    scales: str = "5;23",
    wavelet_scales: str | None = None,
    complex_mode: str = "magnitude",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler: str = "plateau",
    scheduler_patience: int = 3,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    early_stop_patience: int = 0,
    patch_size: int = 16,
    d_model: int = 32,
    heads: int = 4,
    layers: int = 2,
    ff_dim: int = 64,
    dropout: float = 0.0,
    num_workers: int = 0,
    accelerator: str = "auto",
    devices: int | str = "auto",
    train_seed: int = DEFAULT_TRAIN_SEED,
    val_seed: int = DEFAULT_VAL_SEED,
    test_seed: int = DEFAULT_TEST_SEED,
    distinct_train_epochs: bool = False,
    output_dir: str | Path = "outputs/synthetic_engine",
    create_run_dir: bool = True,
) -> tuple[EnginePeakNetwork, L.Trainer, dict[str, Any]]:
    family = family.strip().lower()
    n_outputs = outputs_for_family(family)
    n_train, n_val, n_test = parse_split(split, samples)

    wavelet_list = parse_wavelets(wavelets)
    resolved_wavelet_scales = parse_wavelet_scales(
        wavelet_list, scales, wavelet_scales
    )

    output_path = _unique_run_dir(output_dir) if create_run_dir else Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = dict(
        n_points=n_points,
        x_min=x_min,
        x_max=x_max,
        center_mode=center_mode,
        delta_mu_min=delta_mu_min,
        delta_mu_max=delta_mu_max,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_min_frac=sigma_min_frac,
        sigma_max_frac=sigma_max_frac,
        sigma_ratio_min=sigma_ratio_min,
        sigma_ratio_max=sigma_ratio_max,
        delta_width_min=delta_width_min,
        delta_width_max=delta_width_max,
        amplitude_high_min=amplitude_high_min,
        amplitude_high_max=amplitude_high_max,
        amplitude_ratio_min=amplitude_ratio_min,
        amplitude_ratio_max=amplitude_ratio_max,
        eta_min=eta_min,
        eta_max=eta_max,
        edge_sigma_margin=edge_sigma_margin,
        noise_level=noise_level,
    )
    train_ds, val_ds, test_ds = build_synthetic_datasets(
        family=family,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        train_seed=train_seed,
        val_seed=val_seed,
        test_seed=test_seed,
        distinct_train_epochs=distinct_train_epochs,
        **dataset_kwargs,
    )

    target_mean, target_std = _compute_target_stats(train_ds, n_outputs)

    preprocessing = MultiWaveletPreprocessing(
        resolved_wavelet_scales,
        complex_mode=complex_mode,
    )
    model = EnginePeakNetwork(
        preprocessing=preprocessing,
        signal_length=n_points,
        n_outputs=n_outputs,
        target_mean=target_mean,
        target_std=target_std,
        patch_size=patch_size,
        d_model=d_model,
        nhead=heads,
        num_layers=layers,
        dim_feedforward=ff_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        scheduler=scheduler,
        scheduler_patience=scheduler_patience,
        scheduler_factor=scheduler_factor,
        min_lr=min_lr,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )

    resolved_config = {
        "experiment": "synthetic_peak_engine",
        "generator": "synthetic_peak_generator",
        "family": family,
        "n_outputs": n_outputs,
        "output_names": output_names(n_outputs),
        "samples": samples,
        "split_requested": split,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_points": n_points,
        "x_min": x_min,
        "x_max": x_max,
        "x_step": (x_max - x_min) / (n_points - 1),
        "center_mode": center_mode,
        "center": 0.5 * (x_min + x_max),
        "delta_mu_min": delta_mu_min,
        "delta_mu_max": delta_mu_max,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "sigma_min_frac": sigma_min_frac,
        "sigma_max_frac": sigma_max_frac,
        "sigma_ratio_min": sigma_ratio_min,
        "sigma_ratio_max": sigma_ratio_max,
        "delta_width_min": delta_width_min,
        "delta_width_max": delta_width_max,
        "amplitude_high_min": amplitude_high_min,
        "amplitude_high_max": amplitude_high_max,
        "amplitude_ratio_min": amplitude_ratio_min,
        "amplitude_ratio_max": amplitude_ratio_max,
        "eta_min": eta_min,
        "eta_max": eta_max,
        "edge_sigma_margin": edge_sigma_margin,
        "noise_level": noise_level,
        "input_normalization": False,
        "target_normalization": "train_mean_std_once_then_frozen",
        "wavelets_requested": wavelets,
        "scales_requested": scales,
        "wavelet_scales_requested": wavelet_scales,
        "resolved_wavelet_scales": {
            k: list(v) for k, v in resolved_wavelet_scales.items()
        },
        "complex_mode": complex_mode,
        "preprocessing_name": type(preprocessing).__name__,
        "preprocessing_channels": preprocessing.out_channels,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "scheduler_patience": scheduler_patience,
        "scheduler_factor": scheduler_factor,
        "min_lr": min_lr,
        "early_stop_patience": early_stop_patience,
        "patch_size": patch_size,
        "d_model": d_model,
        "heads": heads,
        "layers": layers,
        "ff_dim": ff_dim,
        "dropout": dropout,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "test_seed": test_seed,
        "distinct_train_epochs": distinct_train_epochs,
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
    }

    model.save_hyperparameters(
        {
            "network_name": "SyntheticPeakEngineTransformer",
            "preprocessing_name": type(preprocessing).__name__,
            **resolved_config,
        },
        ignore=["target_mean", "target_std"],
    )

    # Save exactly what the engine resolved before training starts.
    (output_path / "resolved_config.json").write_text(
        json.dumps(_json_safe(resolved_config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(output_path),
        filename="best-{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks: list[Any] = [checkpoint]
    if early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stop_patience,
            )
        )

    logger = CSVLogger(save_dir=str(output_path), name="", version="")
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(output_path),
    )

    print("\n" + "=" * 78)
    print(" SYNTHETIC PEAK EXPERIMENT ENGINE")
    print(f" run:       {output_path.resolve()}")
    print(f" family:    {family} ({n_outputs} targets)")
    print(f" axis:      {x_min:g}..{x_max:g}, n={n_points}")
    print(f" center:    {center_mode} -> {0.5 * (x_min + x_max):g} when fixed")
    print(f" wavelets:  {resolved_wavelet_scales}")
    print(" input norm: OFF")
    print(f" samples:   {n_train}/{n_val}/{n_test}")
    print("=" * 78 + "\n")

    trainer.fit(model, train_loader, val_loader)

    if checkpoint.best_model_path:
        state = torch.load(
            checkpoint.best_model_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(state["state_dict"], strict=True)

    trainer.test(model, test_loader)

    # Existing project HTML + JSON reporter.
    report = run_generalized_reporting(
        model=model,
        test_loader=test_loader,
        output_dir=output_path,
        train_setup=resolved_config,
        n_outputs=n_outputs,
    )

    summary = {
        "best_checkpoint": checkpoint.best_model_path,
        "report": report,
        "resolved_config": resolved_config,
    }
    (output_path / "engine_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 78)
    print(" EXPERIMENT COMPLETE")
    print(f" best checkpoint: {checkpoint.best_model_path}")
    print(f" mean R2:         {report['mean_r2']:.6f}")
    print(f" report:          {report['report_path']}")
    print(f" metrics:         {output_path / 'metrics.json'}")
    print(f" config:          {output_path / 'resolved_config.json'}")
    print("=" * 78 + "\n")

    return model, trainer, report


def _parse_devices(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    epilog = r"""
SCALE HELP
  5;23;46
      Three explicit scales.

  (21-30;2)
      21,23,25,27,29

  (20;30;5)
      20,25,30

  (20-30;5);(55-100;1)
      Multiple ranges.

PER-WAVELET SCALE HELP
  --wavelets "MH;GD;morl"
  --wavelet-scales "MH=5;23|GD=(1-9;2)|morl=(10-30;5)"

WAVELETS
  Recommended:
      MH, GD, mexh, morl, gaus1, gaus2

  Also supported by PTWT/PyWavelets when installed:
      gaus1..gaus8
      cgau1..cgau8
      cmorB-C      example: cmor1.5-1.0
      shanB-C      example: shan1.5-1.0
      fbspM-B-C    example: fbsp2-1.0-0.5

  Complex wavelets:
      Default representation is magnitude.
      --complex-mode real_imag preserves real and imaginary channels separately.

PRESETS BY EXAMPLE
  Simple default:
      --wavelets "MH" --scales "5;23"

  Dual baseline:
      --wavelets "MH;GD" --scales "5;23"

  Wider multiscale:
      --wavelets "MH;GD;gaus1" --scales "(1-10;1);23"

NOTES
  - No per-spectrum input normalization is performed.
  - Target mean/std are computed once from TRAIN and frozen.
  - Gaussian and Lorentzian use 6 targets.
  - Pseudo-Voigt and mixed families use 8 targets including eta.
  - Existing project report.html and metrics.json are used unchanged.
"""
    p = argparse.ArgumentParser(
        description="Synthetic overlapping-peak experiment engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )

    # Generator.
    p.add_argument(
        "--family",
        choices=["gaussian", "lorentzian", "pseudovoigt", "paired_mix", "independent_mix"],
        default="gaussian",
    )
    p.add_argument("--samples", type=int, default=50_000)
    p.add_argument("--split", type=str, default="0.8;0.1;0.1")
    p.add_argument("--n-points", type=int, default=400)
    p.add_argument("--x-min", type=float, default=0.0)
    p.add_argument("--x-max", type=float, default=400.0)
    p.add_argument("--center-mode", choices=["fixed", "random"], default="fixed")
    p.add_argument("--delta-min", type=float, default=10.0)
    p.add_argument("--delta-max", type=float, default=None)
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--sigma-min-frac", type=float, default=0.02)
    p.add_argument("--sigma-max-frac", type=float, default=0.06)
    p.add_argument("--sigma-ratio-min", type=float, default=0.5)
    p.add_argument("--sigma-ratio-max", type=float, default=2.0)
    p.add_argument("--delta-width-min", type=float, default=0.25)
    p.add_argument("--delta-width-max", type=float, default=3.0)
    p.add_argument("--amp-high-min", type=float, default=0.5)
    p.add_argument("--amp-high-max", type=float, default=1.0)
    p.add_argument("--amp-ratio-min", type=float, default=0.1)
    p.add_argument("--amp-ratio-max", type=float, default=1.0)
    p.add_argument("--eta-min", type=float, default=0.0)
    p.add_argument("--eta-max", type=float, default=1.0)
    p.add_argument("--edge-sigma-margin", type=float, default=4.0)
    p.add_argument("--noise", type=float, default=0.0)

    # Wavelets.
    p.add_argument("--wavelets", type=str, default="MH")
    p.add_argument("--scales", type=str, default="5;23")
    p.add_argument("--wavelet-scales", type=str, default=None)
    p.add_argument(
        "--complex-mode", choices=["magnitude", "real_imag"], default="magnitude"
    )

    # Model/training.
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--scheduler",
        choices=["plateau", "cosine", "none"],
        default="plateau",
    )
    p.add_argument("--scheduler-patience", type=int, default=3)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="0 disables early stopping.",
    )
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ff-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)

    # Runtime/reproducibility.
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--accelerator", type=str, default="auto")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    p.add_argument("--val-seed", type=int, default=DEFAULT_VAL_SEED)
    p.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    p.add_argument("--distinct-train-epochs", action="store_true")
    p.add_argument("--output-dir", type=str, default="outputs/synthetic_engine")
    p.add_argument("--no-timestamp", action="store_true")
    return p


def main() -> None:
    a = build_parser().parse_args()
    run_experiment(
        family=a.family,
        samples=a.samples,
        split=a.split,
        n_points=a.n_points,
        x_min=a.x_min,
        x_max=a.x_max,
        center_mode=a.center_mode,
        delta_mu_min=a.delta_min,
        delta_mu_max=a.delta_max,
        sigma_min=a.sigma_min,
        sigma_max=a.sigma_max,
        sigma_min_frac=a.sigma_min_frac,
        sigma_max_frac=a.sigma_max_frac,
        sigma_ratio_min=a.sigma_ratio_min,
        sigma_ratio_max=a.sigma_ratio_max,
        delta_width_min=a.delta_width_min,
        delta_width_max=a.delta_width_max,
        amplitude_high_min=a.amp_high_min,
        amplitude_high_max=a.amp_high_max,
        amplitude_ratio_min=a.amp_ratio_min,
        amplitude_ratio_max=a.amp_ratio_max,
        eta_min=a.eta_min,
        eta_max=a.eta_max,
        edge_sigma_margin=a.edge_sigma_margin,
        noise_level=a.noise,
        wavelets=a.wavelets,
        scales=a.scales,
        wavelet_scales=a.wavelet_scales,
        complex_mode=a.complex_mode,
        epochs=a.epochs,
        batch_size=a.batch_size,
        lr=a.lr,
        weight_decay=a.weight_decay,
        scheduler=a.scheduler,
        scheduler_patience=a.scheduler_patience,
        scheduler_factor=a.scheduler_factor,
        min_lr=a.min_lr,
        early_stop_patience=a.early_stop_patience,
        patch_size=a.patch_size,
        d_model=a.d_model,
        heads=a.heads,
        layers=a.layers,
        ff_dim=a.ff_dim,
        dropout=a.dropout,
        num_workers=a.num_workers,
        accelerator=a.accelerator,
        devices=_parse_devices(a.devices),
        train_seed=a.train_seed,
        val_seed=a.val_seed,
        test_seed=a.test_seed,
        distinct_train_epochs=a.distinct_train_epochs,
        output_dir=a.output_dir,
        create_run_dir=not a.no_timestamp,
    )


if __name__ == "__main__":
    main()
