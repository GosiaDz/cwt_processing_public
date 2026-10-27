"""Generalized synthetic two-peak datasets for G/L/pseudo-Voigt experiments.

This module is intentionally standalone: it does not modify the existing
``cwt_processing.datasets`` registry.  It implements width-constrained geometry
with a common sigma semantics for Gaussian, Lorentzian and pseudo-Voigt peaks.

Raw target layout for each peak is always::

    [mu, amplitude, sigma, eta]

where ``eta=0`` is Gaussian, ``eta=1`` is Lorentzian and ``eta in [0, 1]`` is
pseudo-Voigt.  ``sigma`` is Gaussian-equivalent sigma; all three families share
FWHM = 2*sqrt(2*ln(2))*sigma.
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

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
_FAMILIES = ("gaussian", "lorentzian", "pseudovoigt")


@dataclass(frozen=True)
class PeakGeometry:
    """One accepted width-constrained geometry sample in grid-point units."""

    mu1_pt: float
    mu2_pt: float
    sigma1_pt: float
    sigma2_pt: float
    amplitude1: float
    amplitude2: float
    family1: str
    family2: str
    eta1: float
    eta2: float


class GeneralizedPeakDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """On-the-fly two-peak generator with width-constrained geometry.

    Geometry follows the broad synthetic setup used in the project:

    * sigma is sampled in grid-point units, by default from 2%-6% of signal length;
    * sigma1/sigma2 must stay inside ``sigma_ratio_min..sigma_ratio_max``;
    * w = mean(FWHM1, FWHM2);
    * delta_mu is sampled from ``max(delta_width_min*w, requested_min)`` to
      ``min(delta_width_max*w, requested_max)``;
    * both peaks must fit inside the signal with configurable sigma margins.

    Samples are deterministic for a given ``seed``, index and epoch.  With
    ``distinct_epochs=True`` only the training dataset needs ``set_epoch(epoch)``
    to obtain a fresh but reproducible sample bank each epoch.
    """

    def __init__(
        self,
        n_samples: int,
        *,
        family: PeakFamily = "gaussian",
        n_points: int = 400,
        x_start: float = 0,
        x_step: float = 1.0,
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
        super().__init__()
        if family not in _FAMILIES + ("paired_mix", "independent_mix"):
            raise ValueError(f"Unknown family '{family}'.")
        if n_points < 32:
            raise ValueError("n_points must be >= 32.")
        if x_step <= 0:
            raise ValueError("x_step must be > 0.")
        if delta_mu_min < 0:
            raise ValueError("delta_mu_min must be >= 0.")
        if delta_mu_max is not None and delta_mu_max < delta_mu_min:
            raise ValueError("delta_mu_max must be >= delta_mu_min.")
        if not (0.0 < sigma_min_frac < sigma_max_frac):
            raise ValueError("Require 0 < sigma_min_frac < sigma_max_frac.")
        if not (0.0 < sigma_ratio_min <= 1.0 <= sigma_ratio_max):
            raise ValueError("sigma ratio bounds must bracket 1.")
        if not (0.0 <= eta_min <= eta_max <= 1.0):
            raise ValueError("eta range must satisfy 0 <= eta_min <= eta_max <= 1.")
        if not (0.0 < amplitude_ratio_min <= amplitude_ratio_max <= 1.0):
            raise ValueError("amplitude ratio must lie in (0, 1].")

        self.n_samples = int(n_samples)
        self.family = family
        self.n_points = int(n_points)
        self.x_start = float(x_start)
        self.x_step = float(x_step)
        self.x_end = self.x_start + (self.n_points - 1) * self.x_step
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
        self._x = self.x_start + np.arange(self.n_points, dtype=np.float64) * self.x_step

    def __len__(self) -> int:
        return self.n_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng(self, index: int) -> np.random.Generator:
        epoch_seed = self.seed + (self._epoch if self.distinct_epochs else 0)
        # SeedSequence prevents accidental correlations between adjacent indices.
        return np.random.default_rng(np.random.SeedSequence([epoch_seed, int(index)]))

    def _sigma_bounds_points(self) -> tuple[float, float]:
        if self.sigma_min is None:
            lo = self.sigma_min_frac * self.n_points
        else:
            lo = self.sigma_min / self.x_step
        if self.sigma_max is None:
            hi = self.sigma_max_frac * self.n_points
        else:
            hi = self.sigma_max / self.x_step
        if not (0.0 < lo < hi):
            raise ValueError(f"Invalid sigma bounds in points: ({lo}, {hi}).")
        return lo, hi

    def _pick_families(self, rng: np.random.Generator) -> tuple[str, str]:
        if self.family in _FAMILIES:
            return self.family, self.family
        if self.family == "paired_mix":
            f = str(rng.choice(_FAMILIES))
            return f, f
        return str(rng.choice(_FAMILIES)), str(rng.choice(_FAMILIES))

    def _eta_for_family(self, family: str, rng: np.random.Generator) -> float:
        if family == "gaussian":
            return 0.0
        if family == "lorentzian":
            return 1.0
        return float(rng.uniform(self.eta_min, self.eta_max))

    def _sample_geometry(self, rng: np.random.Generator) -> PeakGeometry:
        sigma_lo, sigma_hi = self._sigma_bounds_points()
        requested_delta_min_pt = self.delta_mu_min / self.x_step
        requested_delta_max_pt = (
            np.inf if self.delta_mu_max is None else self.delta_mu_max / self.x_step
        )

        for _ in range(self.max_attempts):
            s1 = float(rng.uniform(sigma_lo, sigma_hi))
            s2 = float(rng.uniform(sigma_lo, sigma_hi))
            ratio = s1 / s2
            if not (self.sigma_ratio_min <= ratio <= self.sigma_ratio_max):
                continue

            fwhm1 = FWHM_PER_SIGMA * s1
            fwhm2 = FWHM_PER_SIGMA * s2
            width_ref = 0.5 * (fwhm1 + fwhm2)
            delta_lo = max(self.delta_width_min * width_ref, requested_delta_min_pt)
            delta_hi = min(self.delta_width_max * width_ref, requested_delta_max_pt)
            if delta_hi < delta_lo:
                continue

            delta = float(rng.uniform(delta_lo, delta_hi)) if delta_hi > delta_lo else delta_lo
            center_lo = self.edge_sigma_margin * s1 + 0.5 * delta
            center_hi = (self.n_points - 1) - self.edge_sigma_margin * s2 - 0.5 * delta
            if center_hi <= center_lo:
                continue

            center = float(rng.uniform(center_lo, center_hi))
            mu1_pt = center - 0.5 * delta
            mu2_pt = center + 0.5 * delta

            a_hi = float(rng.uniform(self.amplitude_high_min, self.amplitude_high_max))
            a_ratio = float(rng.uniform(self.amplitude_ratio_min, self.amplitude_ratio_max))
            a_lo = a_hi * a_ratio
            if rng.random() < 0.5:
                a1, a2 = a_hi, a_lo
            else:
                a1, a2 = a_lo, a_hi

            family1, family2 = self._pick_families(rng)
            eta1 = self._eta_for_family(family1, rng)
            eta2 = self._eta_for_family(family2, rng)
            return PeakGeometry(
                mu1_pt=mu1_pt,
                mu2_pt=mu2_pt,
                sigma1_pt=s1,
                sigma2_pt=s2,
                amplitude1=a1,
                amplitude2=a2,
                family1=family1,
                family2=family2,
                eta1=eta1,
                eta2=eta2,
            )

        raise RuntimeError(
            "Could not sample a valid width-constrained pair. "
            "Relax delta_mu/sigma/margin constraints or increase n_points."
        )

    @staticmethod
    def _gaussian(x: np.ndarray, mu: float, amplitude: float, sigma: float) -> np.ndarray:
        t = (x - mu) / sigma
        return amplitude * np.exp(-0.5 * t * t)

    @staticmethod
    def _lorentzian(x: np.ndarray, mu: float, amplitude: float, sigma: float) -> np.ndarray:
        # Parameterized so FWHM = 2*sqrt(2*ln(2))*sigma, same as Gaussian.
        t = (x - mu) / sigma
        return amplitude / (1.0 + (t * t) / (2.0 * np.log(2.0)))

    def _component(
        self,
        family: str,
        x: np.ndarray,
        mu: float,
        amplitude: float,
        sigma: float,
        eta: float,
    ) -> np.ndarray:
        g = self._gaussian(x, mu, amplitude, sigma)
        if family == "gaussian":
            return g
        l = self._lorentzian(x, mu, amplitude, sigma)
        if family == "lorentzian":
            return l
        return (1.0 - eta) * g + eta * l

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = self._rng(index)
        g = self._sample_geometry(rng)

        mu1 = self.x_start + g.mu1_pt * self.x_step
        mu2 = self.x_start + g.mu2_pt * self.x_step
        sigma1 = g.sigma1_pt * self.x_step
        sigma2 = g.sigma2_pt * self.x_step

        y1 = self._component(g.family1, self._x, mu1, g.amplitude1, sigma1, g.eta1)
        y2 = self._component(g.family2, self._x, mu2, g.amplitude2, sigma2, g.eta2)
        signal = y1 + y2

        if self.noise_level > 0.0:
            sigma_noise = self.noise_level * max(float(np.max(signal)), 1e-12)
            signal = signal + rng.normal(0.0, sigma_noise, size=signal.shape)

        targets = np.asarray(
            [
                [mu1, g.amplitude1, sigma1, g.eta1],
                [mu2, g.amplitude2, sigma2, g.eta2],
            ],
            dtype=np.float32,
        )
        return torch.from_numpy(signal.astype(np.float32)), torch.from_numpy(targets)


def flatten_physical_targets(raw_targets: torch.Tensor, n_outputs: int) -> torch.Tensor:
    """Convert (B, 2, 4) [mu,A,sigma,eta] targets to 6- or 8-output vectors."""
    if raw_targets.ndim != 3 or raw_targets.shape[1] != 2 or raw_targets.shape[2] < 4:
        raise ValueError(
            f"Expected raw targets (B,2,4+), got shape {tuple(raw_targets.shape)}."
        )
    p1 = raw_targets[:, 0]
    p2 = raw_targets[:, 1]
    if n_outputs == 6:
        return torch.stack(
            [p1[:, 0], p1[:, 1], p1[:, 2], p2[:, 0], p2[:, 1], p2[:, 2]],
            dim=1,
        )
    if n_outputs == 8:
        return torch.stack(
            [
                p1[:, 0], p1[:, 1], p1[:, 2], p1[:, 3],
                p2[:, 0], p2[:, 1], p2[:, 2], p2[:, 3],
            ],
            dim=1,
        )
    raise ValueError("n_outputs must be 6 or 8.")


def output_names(n_outputs: int) -> list[str]:
    if n_outputs == 6:
        return ["mu1", "A1", "sigma1", "mu2", "A2", "sigma2"]
    if n_outputs == 8:
        return ["mu1", "A1", "sigma1", "eta1", "mu2", "A2", "sigma2", "eta2"]
    raise ValueError("n_outputs must be 6 or 8.")


def get_generalized_peak_datasets(
    *,
    family: PeakFamily,
    n_train: int,
    n_val: int,
    n_test: int,
    train_seed: int = 31337,
    val_seed: int = 7331,
    test_seed: int = 1337,
    distinct_epochs: bool = False,
    **dataset_kwargs,
) -> tuple[GeneralizedPeakDataset, GeneralizedPeakDataset, GeneralizedPeakDataset]:
    """Build deterministic train/validation/test generators with shared scientific settings."""
    train_ds = GeneralizedPeakDataset(
        n_train,
        family=family,
        seed=train_seed,
        distinct_epochs=distinct_epochs,
        **dataset_kwargs,
    )
    val_ds = GeneralizedPeakDataset(
        n_val,
        family=family,
        seed=val_seed,
        distinct_epochs=False,
        **dataset_kwargs,
    )
    test_ds = GeneralizedPeakDataset(
        n_test,
        family=family,
        seed=test_seed,
        distinct_epochs=False,
        **dataset_kwargs,
    )
    return train_ds, val_ds, test_ds
