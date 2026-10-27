"""Select CWT wavelets and scales for one two-peak family.

Examples:
    python cwt_wavelet_scale_selection.py --family gaussian
    python cwt_wavelet_scale_selection.py --family lorentzian
    python cwt_wavelet_scale_selection.py --family pseudovoigt
    
    python cwt_wavelet_scale_selection.py --family gaussian --raport --csv
    python cwt_wavelet_scale_selection.py --family lorentzian --background-limit

CLI parameters:
    --family: Required family (gaussian, lorentzian, pseudovoigt).
    --n-tuning: Tuning geometries (default: 1000).
    --n-test: Independent test geometries (default: 2000).
    --n-null: Pure-noise realizations (default: 1000).
    --output-dir: Output directory (default: current directory).
    --raport / --report: Generate HTML report (default: off).
    --csv: Generate CSV results (default: off).
    --background-limit: Apply the 5% background-peak limit to recommendation
        status (default: off). The background metric is always reported.

The generator is intentionally kept in the separate GeneralizedPeakGen.py file.
This script writes a CSV and a self-contained HTML report, then returns a
result dictionary from ``run_selection`` for use by other Python code.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pywt
from scipy.signal import find_peaks, peak_prominences

from generalized_peak_datasets import GeneralizedPeakDataset


# Experiment defaults follow the project's final selection procedure.
N_TUNING = 1000
N_TEST = 2000
TUNING_SEED = 31337
TEST_SEED = 20261031
N_POINTS = 400
X_START = 1.0
X_STEP = 1.0
SCALES = np.arange(1, 97)
NOISE_LEVELS = (0.00, 0.01, 0.02, 0.03, 0.05)
INTERVAL_TOLERANCES = (0.00, 0.02)
RECOMMENDATION_SUCCESS_TOLERANCE = 0.05
N_NULL = 1000
NULL_SEED = 6012026
NULL_CHUNK_SIZE = 100
FALSE_ALARM_QUANTILE = 0.95
MAX_FALSE_BACKGROUND_SIGNAL_RATE = 0.05
PROMINENCE_MULTIPLIERS = np.asarray(
    (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
)
WAVELETS = ("mexh", "morl") + tuple(f"gaus{i}" for i in range(1, 9))
# WAVELETS = ("mexh", "morl", "gaus1", "gaus2", "gaus3", "gaus4")
PEAK_FAMILIES = ("gaussian", "lorentzian", "pseudovoigt")


@dataclass(frozen=True)
class SelectionConfig:
    family: str
    n_tuning: int = N_TUNING
    n_test: int = N_TEST
    tuning_seed: int = TUNING_SEED
    test_seed: int = TEST_SEED
    scales: tuple[int, ...] = tuple(int(s) for s in SCALES)
    noise_levels: tuple[float, ...] = NOISE_LEVELS
    tolerances: tuple[float, ...] = INTERVAL_TOLERANCES
    n_null: int = N_NULL
    output_dir: Path = Path(".")
    generate_report: bool = False
    generate_csv: bool = False
    apply_background_limit: bool = False


def _make_bank(n_samples: int, seed: int, family: str,
               noise_levels: tuple[float, ...]) -> dict[float, dict[str, np.ndarray]]:
    bank: dict[float, dict[str, np.ndarray]] = {}
    common = dict(
        n_samples=n_samples, family=family, n_points=N_POINTS,
        x_start=X_START, x_step=X_STEP, seed=seed,
    )
    for noise in noise_levels:
        noisy = GeneralizedPeakDataset(noise_level=noise, **common)
        clean = GeneralizedPeakDataset(noise_level=0.0, **common)
        signals, positions, sigmas, input_sigmas = [], [], [], []
        for index in range(n_samples):
            signal, target = noisy[index]
            clean_signal, _ = clean[index]
            target_array = target.numpy()
            clean_array = clean_signal.numpy()
            signals.append(signal.numpy())
            positions.append(np.rint((target_array[:, 0] - X_START) / X_STEP).astype(int))
            sigmas.append(target_array[:, 2] / X_STEP)
            input_sigmas.append(noise * max(float(np.max(clean_array)), 1e-12))
        bank[noise] = {
            "signals": np.asarray(signals), "positions": np.asarray(positions),
            "sigmas": np.asarray(sigmas), "input_sigmas": np.asarray(input_sigmas),
        }
    return bank


def _sample_null_maxima(wavelet: str, scales: np.ndarray, n_samples: int,
                        seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    maxima = np.zeros((n_samples, len(scales)), dtype=float)
    offset = 0
    while offset < n_samples:
        count = min(NULL_CHUNK_SIZE, n_samples - offset)
        noise = rng.normal(size=(count, N_POINTS)).astype(np.float32)
        coefficients, _ = pywt.cwt(noise, scales, wavelet, axis=-1)
        for scale_index, batch in enumerate(coefficients):
            for row_index, row in enumerate(batch):
                peaks, _ = find_peaks(row)
                if peaks.size:
                    maxima[offset + row_index, scale_index] = float(
                        np.max(peak_prominences(row, peaks)[0])
                    )
        offset += count
    return maxima


def _background_mask(n_points: int, positions: np.ndarray,
                     sigmas: np.ndarray) -> np.ndarray:
    x = np.arange(n_points)
    background = np.ones(n_points, dtype=bool)
    for position, sigma in zip(positions, sigmas):
        radius = max(10, int(np.ceil(4.0 * sigma)))
        background &= np.abs(x - position) > radius
    return background


def _score_row(row: np.ndarray, positions: np.ndarray, sigmas: np.ndarray,
               threshold: float) -> tuple[bool, bool, int]:
    peaks, _ = find_peaks(row)
    prominences = peak_prominences(row, peaks)[0] if peaks.size else np.array([])
    accepted = peaks[prominences >= threshold]
    false_count = int(np.count_nonzero(_background_mask(row.size, positions, sigmas)[accepted]))
    left, right = sorted(map(int, positions))
    radius = min(10, max(0, (right - left - 1) // 2))
    if radius < 1:
        return False, false_count > 0, false_count
    success = (np.any(np.abs(accepted - left) <= radius)
               and np.any(np.abs(accepted - right) <= radius))
    return bool(success), false_count > 0, false_count


def _coefficient_noise_sigma(row: np.ndarray, positions: np.ndarray,
                             sigmas: np.ndarray) -> float:
    """Robustly estimate coefficient noise outside the labelled peak supports."""
    background = _background_mask(row.size, positions, sigmas)
    values = row[background]
    if values.size < 16:
        values = row
    median = np.median(values)
    estimate = 1.4826 * np.median(np.abs(values - median))
    return float(max(estimate, np.finfo(float).eps))


def _score_threshold_candidates(
    row: np.ndarray,
    positions: np.ndarray,
    background: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score many prominence thresholds at once for one CWT row."""
    peaks, _ = find_peaks(row)
    prominences = peak_prominences(row, peaks)[0] if peaks.size else np.array([])
    left, right = sorted(map(int, positions))
    radius = min(10, max(0, (right - left - 1) // 2))
    n_candidates = len(thresholds)
    if radius < 1 or not peaks.size:
        return np.zeros(n_candidates, dtype=bool), np.zeros(n_candidates, dtype=bool)

    accepted = prominences[None, :] >= thresholds[:, None]
    left_mask = np.abs(peaks - left) <= radius
    right_mask = np.abs(peaks - right) <= radius
    background_mask = background[peaks]
    success = np.any(accepted[:, left_mask], axis=1) & np.any(accepted[:, right_mask], axis=1)
    false_peak = np.any(accepted[:, background_mask], axis=1)
    return success, false_peak


def _calibrate_threshold_multipliers(
    tuning_bank: dict[float, dict[str, np.ndarray]],
    wavelet: str,
    scales: np.ndarray,
    null_quantiles: np.ndarray,
    noise_levels: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose a prominence multiplier per scale under the 5% background limit."""
    shape = (len(PROMINENCE_MULTIPLIERS), len(noise_levels), len(scales))
    success_counts = np.zeros(shape, dtype=np.int32)
    false_counts = np.zeros(shape, dtype=np.int32)
    sample_counts = np.zeros(len(noise_levels), dtype=np.int32)

    for noise_index, noise in enumerate(noise_levels):
        data = tuning_bank[noise]
        sample_counts[noise_index] = len(data["signals"])
        for signal, positions, sigmas, input_sigma in zip(
            data["signals"], data["positions"], data["sigmas"], data["input_sigmas"]
        ):
            coefficients, _ = pywt.cwt(signal, scales, wavelet)
            for scale_index, row in enumerate(coefficients):
                noise_sigma = _coefficient_noise_sigma(row, positions, sigmas)
                thresholds = np.maximum(
                    null_quantiles[scale_index] * input_sigma,
                    PROMINENCE_MULTIPLIERS * noise_sigma,
                )
                background = _background_mask(row.size, positions, sigmas)
                successes, false_peaks = _score_threshold_candidates(
                    row, positions, background, thresholds
                )
                success_counts[:, noise_index, scale_index] += successes
                false_counts[:, noise_index, scale_index] += false_peaks

    success_rates = success_counts / sample_counts[None, :, None]
    false_rates = false_counts / sample_counts[None, :, None]
    selected_multipliers = np.zeros(len(scales), dtype=float)
    selected_success = np.zeros((len(noise_levels), len(scales)), dtype=float)
    selected_false = np.zeros_like(selected_success)
    for scale_index in range(len(scales)):
        worst_false = np.max(false_rates[:, :, scale_index], axis=1)
        worst_success = np.min(success_rates[:, :, scale_index], axis=1)
        feasible = np.flatnonzero(worst_false <= MAX_FALSE_BACKGROUND_SIGNAL_RATE)
        if feasible.size:
            chosen = max(
                feasible,
                key=lambda i: (worst_success[i], -worst_false[i], -PROMINENCE_MULTIPLIERS[i]),
            )
        else:
            least_false = np.flatnonzero(np.isclose(worst_false, np.min(worst_false)))
            chosen = max(least_false, key=lambda i: (worst_success[i], -PROMINENCE_MULTIPLIERS[i]))
        selected_multipliers[scale_index] = PROMINENCE_MULTIPLIERS[chosen]
        selected_success[:, scale_index] = success_rates[chosen, :, scale_index]
        selected_false[:, scale_index] = false_rates[chosen, :, scale_index]
    return selected_multipliers, selected_success, selected_false


def _evaluate(bank: dict[float, dict[str, np.ndarray]], wavelet: str,
              scales: np.ndarray, null_quantiles: np.ndarray,
              threshold_multipliers: np.ndarray,
              noise_levels: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(noise_levels), len(scales))
    successes, false_rates, false_peaks = (np.zeros(shape, dtype=float) for _ in range(3))
    for noise_index, noise in enumerate(noise_levels):
        data = bank[noise]
        for signal, positions, sigmas, input_sigma in zip(
            data["signals"], data["positions"], data["sigmas"], data["input_sigmas"]
        ):
            coefficients, _ = pywt.cwt(signal, scales, wavelet)
            thresholds = np.zeros(len(scales), dtype=float)
            for scale_index, row in enumerate(coefficients):
                # Re-estimate at each scale; CWT coefficient noise changes with scale.
                thresholds[scale_index] = max(
                    null_quantiles[scale_index] * input_sigma,
                    threshold_multipliers[scale_index]
                    * _coefficient_noise_sigma(row, positions, sigmas),
                )
                success, any_false, false_count = _score_row(
                    row, positions, sigmas, float(thresholds[scale_index])
                )
                successes[noise_index, scale_index] += success
                false_rates[noise_index, scale_index] += any_false
                false_peaks[noise_index, scale_index] += false_count
        n = len(data["signals"])
        successes[noise_index] /= n
        false_rates[noise_index] /= n
        false_peaks[noise_index] /= n
    return successes, false_rates, false_peaks


def _select_interval(scales: np.ndarray, rates: np.ndarray,
                     tolerance: float) -> tuple[int, int, float, float]:
    per_scale_worst = np.min(rates, axis=0)
    best = float(np.max(per_scale_worst))
    eligible = per_scale_worst >= best - tolerance - 1e-12
    candidates = []
    start = 0
    while start < len(scales):
        if not eligible[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(scales) and eligible[end + 1]:
            end += 1
        candidates.append((end - start + 1, float(np.mean(rates[:, start:end + 1])), start, end))
        start = end + 1
    _, _, first, last = max(candidates, key=lambda item: (item[0], item[1]))
    return (int(scales[first]), int(scales[last]), best,
            float(np.min(per_scale_worst[first:last + 1])))


def _recommend_wavelets(
    metrics: dict[str, tuple[float, float]],
    success_tolerance: float = RECOMMENDATION_SUCCESS_TOLERANCE,
    apply_background_limit: bool = False,
) -> tuple[set[str], set[str], float]:
    """Return recommendations based on tuning success and an optional background cap.

    Each metric pair is (worst-case success, background-peak signal rate).
    Success must be within ``success_tolerance`` of the best tuning success.
    Background rate is a secondary acceptability check, not a ranking criterion.
    If enabled, candidates are split by the maximum allowed background-peak rate.
    """
    if not metrics:
        return set(), set(), 0.0
    best_success = max(success for success, _ in metrics.values())
    cutoff = best_success - success_tolerance
    candidates = {
        wavelet: values for wavelet, values in metrics.items()
        if values[0] >= cutoff
    }
    if apply_background_limit:
        recommended = {
            wavelet for wavelet in candidates
            if candidates[wavelet][1] <= MAX_FALSE_BACKGROUND_SIGNAL_RATE
        }
        partially_recommended = set(candidates) - recommended
    else:
        recommended = set(candidates)
        partially_recommended = set()
    return recommended, partially_recommended, cutoff


def _report_html(config: SelectionConfig, rows: list[dict[str, Any]],
                 selections: dict[str, dict[float, tuple[int, int, float, float]]],
                 recommendation_metrics: dict[str, tuple[float, float]],
                 recommended: set[str], partially_recommended: set[str]) -> str:
    palette = ("#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2",
               "#be185d", "#4d7c0f", "#9333ea", "#475569", "#0f766e", "#c026d3")
    colors = {w: palette[i % len(palette)] for i, w in enumerate(WAVELETS)}
    scale_rows, detail_rows, cards, charts, legends = [], [], [], [], []
    for wavelet in WAVELETS:
        legends.append(f'<span><i class="swatch" style="background:{colors[wavelet]}"></i>{escape(wavelet)}</span>')
        for tolerance in config.tolerances:
            start, end, best, floor = selections[wavelet][tolerance]
            interval = str(start) if start == end else f"{start}–{end}"
            scale_rows.append(
                f"<tr class=\"{'recommended-row' if wavelet in recommended else 'partial-row' if wavelet in partially_recommended else ''}\"><td>{escape(wavelet)}</td><td>{tolerance*100:g} p.p.</td>"
                f"<td>{interval}</td><td>{end-start+1}</td><td>{floor:.1%}</td><td>{best:.1%}</td></tr>"
            )
            current = [r for r in rows if r["wavelet"] == wavelet and r["tolerance_pp"] == tolerance * 100]
            for row in current:
                detail_rows.append(
                    f"<tr><td>{escape(wavelet)}</td><td>{tolerance*100:g} p.p.</td><td>{interval}</td>"
                    f"<td>{row['noise']:.0%}</td><td>{row['test_interval_worst_success']:.1%}</td>"
                    f"<td>{row['test_interval_worst_background_signal_rate']:.1%}</td>"
                    f"<td>{row['test_interval_mean_background_peaks_per_signal']:.3f}</td></tr>"
                )
        primary = selections[wavelet][config.tolerances[0]]
        tolerant = selections[wavelet][config.tolerances[-1]]
        if wavelet in recommended:
            status, card_class = '<span class="badge">recommended</span>', "recommended-card"
        elif wavelet in partially_recommended:
            status, card_class = '<span class="partial-badge">partially recommended</span>', "partial-card"
        else:
            status, card_class = "", ""
        tuning_success, tuning_background = recommendation_metrics[wavelet]
        primary_rows = [r for r in rows if r["wavelet"] == wavelet
                        and r["tolerance_pp"] == config.tolerances[0] * 100]
        performance_rows = []
        for noise in (0.00, 0.01, 0.05):
            result = next((r for r in primary_rows if np.isclose(r["noise"], noise)), None)
            if result is not None:
                performance_rows.append(
                    f'<div class="performance-row"><span>Success at {noise:.0%} noise</span>'
                    f'<strong>{result["test_interval_worst_success"]:.1%}</strong></div>'
                )
        test_at_five = next((r for r in primary_rows if np.isclose(r["noise"], 0.05)), None)
        if test_at_five is not None:
            performance_rows.append(
                '<div class="performance-row"><span>Signals with a background peak at 5% noise</span>'
                f'<strong>{test_at_five["test_interval_worst_background_signal_rate"]:.1%}</strong></div>'
            )
        performance = (f'<div class="performance">{ "".join(performance_rows) }</div>'
                       if performance_rows else "")
        cards.append(
            f'<article class="{card_class}"><h3>{escape(wavelet)} {status}</h3><p>Scale at 0 pp tolerance: '
            f'<strong>{primary[0]}–{primary[1]}</strong></p><p>Scales range at '
            f'{config.tolerances[-1]*100:g} pp tolerance: <strong>{tolerant[0]}–{tolerant[1]}</strong></p>'
            f'<p>Tuning: worst-case success <strong>{tuning_success:.1%}</strong>; '
            f'signals with a background peak at 0–5% noise <strong>{tuning_background:.1%}</strong></p>'
            f'<h4>Independent test performance</h4>{performance}</article>'
        )
    width, height, left, right, top, bottom = 820, 430, 70, 25, 35, 65
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [left + i * plot_w / max(1, len(config.noise_levels) - 1)
          for i in range(len(config.noise_levels))]
    for tolerance in config.tolerances:
        parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Test success rate">',
                 '<rect width="100%" height="100%" fill="white"/>']
        for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
            y = top + (0.85 - tick) / 0.85 * plot_h
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end">{tick:.0%}</text>')
        for i, noise in enumerate(config.noise_levels):
            parts.append(f'<text x="{xs[i]:.1f}" y="{height-35}" text-anchor="middle">{noise:.0%}</text>')
        parts.extend([f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#374151"/>',
                      f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#374151"/>',
                      f'<text x="{width/2}" y="{height-8}" text-anchor="middle">Noise level</text>',
                      f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle">Worst success across selected scales</text>'])
        for wavelet in WAVELETS:
            selected = sorted((r for r in rows if r["wavelet"] == wavelet
                               and r["tolerance_pp"] == tolerance * 100), key=lambda r: r["noise"])
            values = [r["test_interval_worst_success"] for r in selected]
            points = " ".join(f"{xs[i]:.1f},{top + (0.85-values[i])/0.85*plot_h:.1f}"
                               for i in range(len(values)))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[wavelet]}" stroke-width="2.5"/>')
        parts.append("</svg>")
        charts.append(f"<h3>Tolerance {tolerance*100:g} pp</h3>{''.join(parts)}")
    family_label = {"gaussian": "Gaussian", "lorentzian": "Lorentzian", "pseudovoigt": "pseudo-Voigt"}[config.family]
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CWT selection — {family_label}</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#18212f;max-width:1120px;margin:32px auto;padding:0 22px}}h1,h2{{line-height:1.2}}.muted{{color:#586579}}.panel{{background:#f5f7fa;padding:16px 20px;border-radius:10px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 28px}}th,td{{padding:9px 10px;border-bottom:1px solid #dce2ea;text-align:right}}th{{background:#f2f4f7}}th:first-child,td:first-child{{text-align:left}}svg{{width:100%;height:auto;font:13px system-ui,sans-serif}}.legend{{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0 20px}}.swatch{{display:inline-block;width:20px;height:4px;vertical-align:middle;margin-right:6px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}article{{border:1px solid #cbd5e1;border-radius:10px;padding:12px 18px;background:#f8fafc}}article h4{{margin:14px 0 4px}}.performance-row{{display:flex;justify-content:space-between;gap:12px;padding:4px 0;border-bottom:1px solid #dce2ea}}.recommended-card{{border:2px solid #16856b;background:#effaf6}}.partial-card{{border:2px solid #d97706;background:#fffaf0}}.badge,.partial-badge{{font-size:12px;vertical-align:middle;border-radius:999px;padding:4px 9px}}.badge{{background:#d2f1e6;color:#075e4b}}.partial-badge{{background:#ffedc2;color:#854d0e}}.recommended-row td{{background:#f0faf6;font-weight:650}}.recommended-row td:first-child{{border-left:4px solid #16856b}}.partial-row td{{background:#fffaf0;font-weight:650}}.partial-row td:first-child{{border-left:4px solid #d97706}}code{{background:#f1f3f5;padding:2px 5px;border-radius:4px}}</style></head><body>
<h1>CWT wavelet and scale selection for resolving two {family_label} peaks</h1><p class="muted">Independent test · family: <code>{config.family}</code> · wavelets: {', '.join(escape(w) for w in WAVELETS)}</p>
<div class="panel">Scales are selected using the tuning set only. Success means finding two distinct local CWT maxima, one within the tolerance window around each true peak. Additional maxima outside those windows are reported as background peaks.</div>
<h2>Recommended wavelets and scales</h2><p>Recommendations are determined using the tuning set only. Wavelets more than {RECOMMENDATION_SUCCESS_TOLERANCE:.0%} below the best worst-case success rate are excluded. The background-peak rate is a secondary diagnostic and does not rank or eliminate candidates by small differences. {'The 5% background-peak signal-rate limit is applied: candidates within the limit are fully recommended, and those above it are partially recommended.' if config.apply_background_limit else 'The optional 5% background-peak signal-rate limit is disabled; all candidates within the success range are recommended.'} The background rate is the worst observed over scales in the selected 0 pp range and noise levels from 0% to 5%. Values on the cards refer to the selected range at 0 pp tolerance.</p><div class="cards">{''.join(cards)}</div>
<h2>Selected scales and ranges</h2><table><thead><tr><th>Wavelet</th><th>Tolerance</th><th>Scale range</th><th>Number of scales</th><th>Worst tuning success</th><th>Best tuning success</th></tr></thead><tbody>{''.join(scale_rows)}</tbody></table>
<h2>Performance on the independent test set</h2><p>The charts show the worst success rate across scales in the selected range at each noise level.</p><div class="legend">{''.join(legends)}</div>{''.join(charts)}
<h2>Results by noise level</h2><table><thead><tr><th>Wavelet</th><th>Tolerance</th><th>Range</th><th>Noise</th><th>Worst success</th><th>Signals with a background peak</th><th>Mean background peaks / signal</th></tr></thead><tbody>{''.join(detail_rows)}</tbody></table>
<h2>Method</h2><ul><li>Family: {escape(config.family)}; scales: {config.scales[0]}–{config.scales[-1]}; noise levels: {', '.join(f'{n:.0%}' for n in config.noise_levels)}.</li><li>For each wavelet and scale, prominence thresholds are calibrated on the tuning set. Candidates combine the 95th percentile pure-noise null threshold with a robust coefficient-noise estimate from outside the peak supports, multiplied by a searched prominence factor. Threshold calibration considers the {MAX_FALSE_BACKGROUND_SIGNAL_RATE:.0%} background-rate target; this target is {'also applied' if config.apply_background_limit else 'not applied'} when assigning recommendation status.</li><li>Tuning set: {config.n_tuning} geometries, seed {config.tuning_seed}. Independent test set: {config.n_test} geometries, seed {config.test_seed}.</li><li>Recommendation selection uses tuning data only. Candidates must be within {RECOMMENDATION_SUCCESS_TOLERANCE:.0%} of the highest worst-case success rate. Background-peak rate is shown as a secondary metric and is used as a status limit only when enabled; the test set is not used to select recommendations.</li><li>At 0 pp tolerance, the best worst-case success is selected. At other tolerances, the widest continuous scale range within the allowed success-rate drop is selected.</li></ul>
<p><strong>Interpretation:</strong> the null threshold controls false alarms on pure noise. The background peak rate on signals can differ because of interactions between the signal and wavelet.</p><p>Configuration and full numeric results are also saved in the CSV file.</p></body></html>'''


def run_selection(config: SelectionConfig) -> dict[str, Any]:
    """Run all configured wavelets, save CSV/HTML, and return results and paths."""
    if config.family not in PEAK_FAMILIES:
        raise ValueError(
            f"Invalid peak family {config.family!r}. Choose one of: "
            f"{', '.join(PEAK_FAMILIES)}."
        )
    for wavelet in WAVELETS:
        pywt.ContinuousWavelet(wavelet)  # fail early with a clear PyWavelets error
    scales = np.asarray(config.scales, dtype=int)
    if scales.ndim != 1 or len(scales) == 0 or np.any(scales <= 0):
        raise ValueError("scales must be a non-empty sequence of positive integers")
    stem = f"cwt_selection_{config.family}"
    csv_path = config.output_dir / f"{stem}_results.csv"
    html_path = config.output_dir / f"{stem}_report.html"
    if config.generate_csv or config.generate_report:
        config.output_dir.mkdir(parents=True, exist_ok=True)
    print("Preparing matched tuning and test sets...", flush=True)
    tuning = _make_bank(config.n_tuning, config.tuning_seed, config.family, config.noise_levels)
    test = _make_bank(config.n_test, config.test_seed, config.family, config.noise_levels)
    rows: list[dict[str, Any]] = []
    selections: dict[str, dict[float, tuple[int, int, float, float]]] = {}
    recommendation_metrics: dict[str, tuple[float, float]] = {}
    for wavelet_index, wavelet in enumerate(WAVELETS):
        print(f"Calibrating threshold and selecting scales: {wavelet}", flush=True)
        null_maxima = _sample_null_maxima(wavelet, scales, config.n_null,
                                          NULL_SEED + wavelet_index)
        null_quantiles = np.quantile(null_maxima, FALSE_ALARM_QUANTILE, axis=0)
        threshold_multipliers, tuning_success, tuning_false = _calibrate_threshold_multipliers(
            tuning, wavelet, scales, null_quantiles, config.noise_levels
        )
        selected = {tol: _select_interval(scales, tuning_success, tol) for tol in config.tolerances}
        selections[wavelet] = selected
        strict_start, strict_end, _, _ = selected[config.tolerances[0]]
        strict_indices = np.flatnonzero((scales >= strict_start) & (scales <= strict_end))
        recommendation_metrics[wavelet] = (
            float(np.min(tuning_success[:, strict_indices])),
            float(np.max(tuning_false[:, strict_indices])),
        )
        selected_scales = np.asarray(sorted({s for a, b, _, _ in selected.values()
                                             for s in range(a, b + 1)}), dtype=int)
        test_success, test_false, test_peaks = _evaluate(
            test, wavelet, selected_scales,
            null_quantiles[np.searchsorted(scales, selected_scales)],
            threshold_multipliers[np.searchsorted(scales, selected_scales)],
            config.noise_levels,
        )
        indices = {int(scale): i for i, scale in enumerate(selected_scales)}
        for tolerance, (start, end, best, floor) in selected.items():
            idx = [indices[s] for s in range(start, end + 1)]
            for noise_index, noise in enumerate(config.noise_levels):
                rows.append({
                    "family": config.family, "wavelet": wavelet,
                    "tolerance_pp": tolerance * 100, "scale_start": start, "scale_end": end,
                    "width_scales": end - start + 1,
                    "tuning_best_worst_success": best,
                    "tuning_interval_worst_success": floor, "noise": noise,
                    "test_interval_worst_success": float(np.min(test_success[noise_index, idx])),
                    "test_interval_worst_background_signal_rate": float(np.max(test_false[noise_index, idx])),
                    "test_interval_mean_background_peaks_per_signal": float(np.mean(test_peaks[noise_index, idx])),
                    "n_tuning": config.n_tuning, "tuning_seed": config.tuning_seed,
                    "n_test": config.n_test, "test_seed": config.test_seed,
                })
    if config.generate_csv:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    recommended, partially_recommended, recommendation_success_cutoff = _recommend_wavelets(
        recommendation_metrics, apply_background_limit=config.apply_background_limit
    )
    if config.generate_report:
        html_path.write_text(
            _report_html(
                config, rows, selections, recommendation_metrics,
                recommended, partially_recommended,
            ),
            encoding="utf-8",
        )
    fully_recommended_wavelets = sorted(recommended)
    partially_recommended_wavelets = sorted(partially_recommended)
    recommended_scales = {
        wavelet: list(range(selections[wavelet][config.tolerances[0]][0],
                            selections[wavelet][config.tolerances[0]][1] + 1))
        for wavelet in fully_recommended_wavelets
    }
    recommended_scale_ranges = {
        wavelet: {
            tolerance * 100: {
                "start": selections[wavelet][tolerance][0],
                "end": selections[wavelet][tolerance][1],
            }
            for tolerance in config.tolerances
        }
        for wavelet in fully_recommended_wavelets
    }
    partially_recommended_scales = {
        wavelet: list(range(selections[wavelet][config.tolerances[0]][0],
                            selections[wavelet][config.tolerances[0]][1] + 1))
        for wavelet in partially_recommended_wavelets
    }
    partially_recommended_scale_ranges = {
        wavelet: {
            tolerance * 100: {
                "start": selections[wavelet][tolerance][0],
                "end": selections[wavelet][tolerance][1],
            }
            for tolerance in config.tolerances
        }
        for wavelet in partially_recommended_wavelets
    }
    print("\nSelections on the tuning set")
    for wavelet, per_tolerance in selections.items():
        for tolerance, (start, end, best, floor) in per_tolerance.items():
            print(f"{wavelet:8s} | tolerance {tolerance:.0%} | scales {start}–{end} | tuning {floor:.3f}/{best:.3f}")
    print(f"Recommendation success cutoff: {recommendation_success_cutoff:.1%}")
    print(f"Background-rate recommendation limit: {'enabled' if config.apply_background_limit else 'disabled'}")
    print("Recommended: " + ", ".join(fully_recommended_wavelets))
    if config.apply_background_limit:
        print("Partially recommended (background rate above limit): " + ", ".join(partially_recommended_wavelets))
    saved_files = [str(path) for enabled, path in (
        (config.generate_csv, csv_path), (config.generate_report, html_path)
    ) if enabled]
    print("Saved " + (", ".join(saved_files) if saved_files else "no output files"))
    return {"config": config, "selections": selections,
            "recommendation_metrics": recommendation_metrics,
            "recommendation_success_cutoff": recommendation_success_cutoff,
            "max_background_signal_rate": MAX_FALSE_BACKGROUND_SIGNAL_RATE,
            "background_limit_enabled": config.apply_background_limit,
            "recommended_wavelets": fully_recommended_wavelets,
            "partially_recommended_wavelets": partially_recommended_wavelets,
            "recommended_scales": recommended_scales,
            "recommended_scale_ranges": recommended_scale_ranges,
            "partially_recommended_scales": partially_recommended_scales,
            "partially_recommended_scale_ranges": partially_recommended_scale_ranges}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True,
                        choices=PEAK_FAMILIES,
                        help="Peak family used for both peaks")
    parser.add_argument("--n-tuning", type=int, default=N_TUNING)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--n-null", type=int, default=N_NULL)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--raport", "--report", dest="report", action="store_true",
                        help="Generate the HTML report (disabled by default)")
    parser.add_argument("--csv", action="store_true", default=False,
                        help="Generate the CSV file (disabled by default)")
    parser.add_argument("--background-limit", action="store_true", default=False,
                        help="Apply the 5% background-peak rate limit to recommendation status (disabled by default)")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = _parse_args()
    if min(args.n_tuning, args.n_test, args.n_null) < 1:
        raise SystemExit("Sample counts must be positive")
    return run_selection(SelectionConfig(
        family=args.family, n_tuning=args.n_tuning,
        n_test=args.n_test, n_null=args.n_null, output_dir=args.output_dir,
        generate_report=args.report, generate_csv=args.csv,
        apply_background_limit=args.background_limit,
    ))


if __name__ == "__main__":
    main()

