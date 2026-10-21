
"""Synthetic theoretical peak generator for the experiment engine.

This file is intentionally standalone and does not modify the existing dataset
registry.  It generates two overlapping synthetic peaks with a fixed pair center
by default.

Raw target shape:
    (2, 4) = [[mu1, A1, sigma1, eta1],
              [mu2, A2, sigma2, eta2]]

Families:
    gaussian
    lorentzian
    pseudovoigt
    paired_mix
    independent_mix

For Gaussian eta=0; for Lorentzian eta=1; for pseudo-Voigt eta is sampled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

PeakFamily = Literal[
    "gaussian",
    "lorentzian",
    "pseudovoigt",
    "paired_mix",
    "independent_mix",
]

BASE_FAMILIES = ("gaussian", "lorentzian", "pseudovoigt")
FWHM_PER_SIGMA = 2.3548200450309493


@dataclass(frozen=True)
class PeakGeometry:
    mu1: float
    mu2: float
    sigma1: float
    sigma2: float
    amplitude1: float
    amplitude2: float
    family1: str
    family2: str
    eta1: float
    eta2: float


def outputs_for_family(family: str) -> int:
    """Return the natural regression head size for the selected family."""
    family = family.strip().lower()
    if family in {"gaussian", "lorentzian"}:
        return 6
    if family in {"pseudovoigt", "paired_mix", "independent_mix"}:
        return 8
    raise ValueError(f"Unknown peak family: {family!r}")


def flatten_targets(raw_targets: torch.Tensor, n_outputs: int) -> torch.Tensor:
    """Flatten (..., 2, 4) physical targets to (..., 6) or (..., 8)."""
    if raw_targets.shape[-2:] != (2, 4):
        raise ValueError(f"Expected (...,2,4) targets, got {tuple(raw_targets.shape)}.")
    p1 = raw_targets[..., 0, :]
    p2 = raw_targets[..., 1, :]
    if n_outputs == 6:
        return torch.stack(
            [p1[..., 0], p1[..., 1], p1[..., 2],
             p2[..., 0], p2[..., 1], p2[..., 2]],
            dim=-1,
        )
    if n_outputs == 8:
        return torch.stack(
            [p1[..., 0], p1[..., 1], p1[..., 2], p1[..., 3],
             p2[..., 0], p2[..., 1], p2[..., 2], p2[..., 3]],
            dim=-1,
        )
    raise ValueError("n_outputs must be 6 or 8.")


def output_names(n_outputs: int) -> list[str]:
    if n_outputs == 6:
        return ["mu1", "A1", "sigma1", "mu2", "A2", "sigma2"]
    if n_outputs == 8:
        return ["mu1", "A1", "sigma1", "eta1", "mu2", "A2", "sigma2", "eta2"]
    raise ValueError("n_outputs must be 6 or 8.")


class SyntheticPeakDataset(Dataset):
    """Deterministic lazy generator of two overlapping synthetic peaks.

    Default geometry:
      - axis: 0..400 (inclusive)
      - 400 points (resolution can be changed with n_points)
      - pair center fixed at the exact axis midpoint
      - no input normalization
      - no noise by default

    Width fractions are defined against the physical axis span, not point count,
    so changing n_points changes resolution without changing the physical problem.
    """

    def __init__(
        self,
        n_samples: int,
        *,
        family: PeakFamily = "gaussian",
        n_points: int = 400,
        x_min: float = 0.0,
        x_max: float = 400.0,
        center_mode: Literal["fixed", "random"] = "fixed",
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
        seed: int = 31337,
        distinct_epochs: bool = False,
        max_attempts: int = 10_000,
    ) -> None:
        family = family.strip().lower()
        if family not in {"gaussian", "lorentzian", "pseudovoigt", "paired_mix", "independent_mix"}:
            raise ValueError(f"Unsupported family: {family!r}")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if n_points < 32:
            raise ValueError("n_points must be >= 32.")
        if not x_max > x_min:
            raise ValueError("Require x_max > x_min.")
        if center_mode not in {"fixed", "random"}:
            raise ValueError("center_mode must be 'fixed' or 'random'.")
        if not (0.0 < sigma_min_frac < sigma_max_frac):
            raise ValueError("Require 0 < sigma_min_frac < sigma_max_frac.")
        if not (0.0 < sigma_ratio_min <= 1.0 <= sigma_ratio_max):
            raise ValueError("sigma ratio bounds must bracket 1.")
        if not (0.0 < amplitude_ratio_min <= amplitude_ratio_max <= 1.0):
            raise ValueError("Amplitude ratio bounds must lie in (0,1].")
        if not (0.0 <= eta_min <= eta_max <= 1.0):
            raise ValueError("Require 0 <= eta_min <= eta_max <= 1.")
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative.")

        self.n_samples = int(n_samples)
        self.family = family
        self.n_points = int(n_points)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.x_start = self.x_min  # reporter compatibility
        self.x_step = (self.x_max - self.x_min) / (self.n_points - 1)
        self.center_mode = center_mode
        self.center = 0.5 * (self.x_min + self.x_max)

        self.delta_mu_min = float(delta_mu_min)
        self.delta_mu_max = None if delta_mu_max is None else float(delta_mu_max)
        self.sigma_min = None if sigma_min is None else float(sigma_min)
        self.sigma_max = None if sigma_max is None else float(sigma_max)
        self.sigma_min_frac = float(sigma_min_frac)
        self.sigma_max_frac = float(sigma_max_frac)
        self.sigma_ratio_min = float(sigma_ratio_min)
        self.sigma_ratio_max = float(sigma_ratio_max)
        self.delta_width_min = float(delta_width_min)
        self.delta_width_max = float(delta_width_max)
        self.amplitude_high_min = float(amplitude_high_min)
        self.amplitude_high_max = float(amplitude_high_max)
        self.amplitude_ratio_min = float(amplitude_ratio_min)
        self.amplitude_ratio_max = float(amplitude_ratio_max)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.edge_sigma_margin = float(edge_sigma_margin)
        self.noise_level = float(noise_level)
        self.seed = int(seed)
        self.distinct_epochs = bool(distinct_epochs)
        self.max_attempts = int(max_attempts)
        self._epoch = 0
        self._x = np.linspace(self.x_min, self.x_max, self.n_points, dtype=np.float64)

    def __len__(self) -> int:
        return self.n_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        epoch_seed = self.seed + (self._epoch if self.distinct_epochs else 0)
        return np.random.default_rng(np.random.SeedSequence([epoch_seed, int(index)]))

    def _sigma_bounds(self) -> tuple[float, float]:
        span = self.x_max - self.x_min
        lo = self.sigma_min if self.sigma_min is not None else self.sigma_min_frac * span
        hi = self.sigma_max if self.sigma_max is not None else self.sigma_max_frac * span
        if not (0.0 < lo < hi):
            raise ValueError(f"Invalid sigma bounds: ({lo}, {hi}).")
        return float(lo), float(hi)

    def _pick_families(self, rng: np.random.Generator) -> tuple[str, str]:
        if self.family in BASE_FAMILIES:
            return self.family, self.family
        if self.family == "paired_mix":
            f = str(rng.choice(BASE_FAMILIES))
            return f, f
        return str(rng.choice(BASE_FAMILIES)), str(rng.choice(BASE_FAMILIES))

    def _eta_for_family(self, family: str, rng: np.random.Generator) -> float:
        if family == "gaussian":
            return 0.0
        if family == "lorentzian":
            return 1.0
        return float(rng.uniform(self.eta_min, self.eta_max))

    def _sample_geometry(self, rng: np.random.Generator) -> PeakGeometry:
        sigma_lo, sigma_hi = self._sigma_bounds()

        for _ in range(self.max_attempts):
            s1 = float(rng.uniform(sigma_lo, sigma_hi))
            s2 = float(rng.uniform(sigma_lo, sigma_hi))
            ratio = s1 / s2
            if not (self.sigma_ratio_min <= ratio <= self.sigma_ratio_max):
                continue

            fwhm1 = FWHM_PER_SIGMA * s1
            fwhm2 = FWHM_PER_SIGMA * s2
            width_ref = 0.5 * (fwhm1 + fwhm2)

            delta_lo = max(self.delta_mu_min, self.delta_width_min * width_ref)
            delta_hi = self.delta_width_max * width_ref
            if self.delta_mu_max is not None:
                delta_hi = min(delta_hi, self.delta_mu_max)
            if delta_hi < delta_lo:
                continue

            delta = float(rng.uniform(delta_lo, delta_hi)) if delta_hi > delta_lo else delta_lo

            if self.center_mode == "fixed":
                center = self.center
            else:
                center_lo = self.x_min + self.edge_sigma_margin * s1 + 0.5 * delta
                center_hi = self.x_max - self.edge_sigma_margin * s2 - 0.5 * delta
                if center_hi <= center_lo:
                    continue
                center = float(rng.uniform(center_lo, center_hi))

            mu1 = center - 0.5 * delta
            mu2 = center + 0.5 * delta

            if mu1 - self.edge_sigma_margin * s1 < self.x_min:
                continue
            if mu2 + self.edge_sigma_margin * s2 > self.x_max:
                continue

            a_hi = float(rng.uniform(self.amplitude_high_min, self.amplitude_high_max))
            a_lo = a_hi * float(rng.uniform(self.amplitude_ratio_min, self.amplitude_ratio_max))
            if rng.random() < 0.5:
                a1, a2 = a_hi, a_lo
            else:
                a1, a2 = a_lo, a_hi

            family1, family2 = self._pick_families(rng)
            eta1 = self._eta_for_family(family1, rng)
            eta2 = self._eta_for_family(family2, rng)

            return PeakGeometry(mu1, mu2, s1, s2, a1, a2, family1, family2, eta1, eta2)

        raise RuntimeError(
            "Could not sample a valid fixed-center pair. Relax width/delta/margin "
            "constraints or widen the signal axis."
        )

    @staticmethod
    def _gaussian(x: np.ndarray, mu: float, amplitude: float, sigma: float) -> np.ndarray:
        t = (x - mu) / sigma
        return amplitude * np.exp(-0.5 * t * t)

    @staticmethod
    def _lorentzian(x: np.ndarray, mu: float, amplitude: float, sigma: float) -> np.ndarray:
        # Same sigma convention as the project: Gaussian-equivalent FWHM parameter.
        t = (x - mu) / sigma
        return amplitude / (1.0 + (t * t) / (2.0 * np.log(2.0)))

    def _component(
        self, family: str, mu: float, amplitude: float, sigma: float, eta: float
    ) -> np.ndarray:
        g = self._gaussian(self._x, mu, amplitude, sigma)
        if family == "gaussian":
            return g
        l = self._lorentzian(self._x, mu, amplitude, sigma)
        if family == "lorentzian":
            return l
        return (1.0 - eta) * g + eta * l

    def targets_only(self, index: int) -> torch.Tensor:
        """Generate targets without constructing the full signal."""
        rng = self._rng(index)
        g = self._sample_geometry(rng)
        arr = np.asarray(
            [
                [g.mu1, g.amplitude1, g.sigma1, g.eta1],
                [g.mu2, g.amplitude2, g.sigma2, g.eta2],
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(arr)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng(index)
        g = self._sample_geometry(rng)

        y1 = self._component(g.family1, g.mu1, g.amplitude1, g.sigma1, g.eta1)
        y2 = self._component(g.family2, g.mu2, g.amplitude2, g.sigma2, g.eta2)
        signal = y1 + y2

        if self.noise_level > 0.0:
            sigma_noise = self.noise_level * max(float(np.max(signal)), 1e-12)
            signal = signal + rng.normal(0.0, sigma_noise, size=signal.shape)

        targets = np.asarray(
            [
                [g.mu1, g.amplitude1, g.sigma1, g.eta1],
                [g.mu2, g.amplitude2, g.sigma2, g.eta2],
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(signal.astype(np.float32)), torch.from_numpy(targets)


def build_synthetic_datasets(
    *,
    family: PeakFamily,
    n_train: int,
    n_val: int,
    n_test: int,
    train_seed: int = 31337,
    val_seed: int = 7331,
    test_seed: int = 1337,
    distinct_train_epochs: bool = False,
    **kwargs,
) -> tuple[SyntheticPeakDataset, SyntheticPeakDataset, SyntheticPeakDataset]:
    """Build deterministic train/validation/test datasets."""
    train = SyntheticPeakDataset(
        n_train, family=family, seed=train_seed,
        distinct_epochs=distinct_train_epochs, **kwargs
    )
    val = SyntheticPeakDataset(
        n_val, family=family, seed=val_seed,
        distinct_epochs=False, **kwargs
    )
    test = SyntheticPeakDataset(
        n_test, family=family, seed=test_seed,
        distinct_epochs=False, **kwargs
    )
    return train, val, test
