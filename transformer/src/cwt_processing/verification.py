"""Comprehensive network verification, evaluation metrics, and HTML report generator."""

from __future__ import annotations
import base64
import csv
from datetime import datetime
import io
from html import escape
import json
import re
from pathlib import Path
import subprocess
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader
from cwt_processing.networks.base import BaseSpectrumNetwork

def load_loss_history_from_csv(csv_path: str | Path) -> dict[str, list[float]]:
    """Parse epoch-level train_loss and val_loss from Lightning's metrics.csv."""
    path = Path(csv_path)
    if not path.exists():
        return {"epochs": [], "train_loss": [], "val_loss": []}
    train_losses: dict[int, float] = {}
    val_losses: dict[int, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "epoch" not in row or row["epoch"] == "":
                continue
            epoch = int(row["epoch"])
            if row.get("train_loss"):
                train_losses[epoch] = float(row["train_loss"])
            if row.get("val_loss"):
                val_losses[epoch] = float(row["val_loss"])
    all_epochs = sorted(set(train_losses.keys()) | set(val_losses.keys()))
    epochs = [e + 1 for e in all_epochs]
    train_loss = [train_losses[e] for e in all_epochs if e in train_losses]
    val_loss = [val_losses[e] for e in all_epochs if e in val_losses]
    return {"epochs": epochs, "train_loss": train_loss, "val_loss": val_loss}


def get_git_info() -> dict[str, str]:
    """Retrieve Git commit hash, remote URL / repository path, and active branch."""
    try:
        githash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        githash = "unknown"
    try:
        gitpath = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        try:
            gitpath = subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            gitpath = str(Path.cwd())
    try:
        gitbranch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        gitbranch = "unknown"
    return {"githash": githash, "gitpath": gitpath, "gitbranch": gitbranch}


def unpack_predictions_and_targets(
    decoded: np.ndarray,
    raw_targets: np.ndarray,
    sort_peaks: bool = True,
) -> dict[str, Any]:
    """Align and unpack decoded predictions and ground truth targets into standard arrays.

    Returns dictionary containing:

        - pred_locs: (N, K)

        - true_locs: (N, K)

        - pred_heights: (N, K)

        - true_heights: (N, K)

        - pred_widths: (N, K) or None

        - true_widths: (N, K) or None

    """
    n_samples = decoded.shape[0]
    # Ground truth targets: (N, K, 5) -> [c, amplitude, location, scale, skew]
    if raw_targets.ndim == 3 and raw_targets.shape[-1] >= 4:
        true_locs = raw_targets[:, :, 2]
        true_heights = raw_targets[:, :, 1]
        # Gaussian HWHM (half width at half maximum): sqrt(2 * ln(2)) * sigma
        true_scales = raw_targets[:, :, 3]
        true_widths = np.sqrt(2.0 * np.log(2.0)) * true_scales
    elif raw_targets.ndim == 3 and raw_targets.shape[-1] == 2:
        true_locs = raw_targets[:, :, 0]
        true_heights = raw_targets[:, :, 1]
        true_widths = None
    else:
        true_locs = raw_targets[:, [0, 2]] if raw_targets.shape[1] >= 4 else raw_targets[:, :1]
        true_heights = raw_targets[:, [1, 3]] if raw_targets.shape[1] >= 4 else raw_targets[:, :1]
        true_widths = None
    k_peaks = true_locs.shape[1] if true_locs.ndim == 2 else 1
    # Unpack predicted locations, heights, and half-widths
    pred_widths = None
    pred_etas = None
    true_etas = raw_targets[:, :, 4] if decoded.shape[1] == 8 and raw_targets.ndim == 3 and raw_targets.shape[-1] >= 5 else None
    if decoded.ndim == 2 and decoded.shape[1] == 4:
        pred_locs = decoded[:, [0, 2]]
        pred_heights = decoded[:, [1, 3]]
    elif decoded.ndim == 2 and decoded.shape[1] == 8:
        pred_locs = decoded[:, [0, 4]].copy()
        pred_heights = decoded[:, [1, 5]].copy()
        pred_widths = np.sqrt(2.0 * np.log(2.0)) * decoded[:, [2, 6]]
        pred_etas = decoded[:, [3, 7]].copy()
    elif decoded.ndim == 2 and decoded.shape[1] == 6:
        pred_locs = decoded[:, [0, 3]]
        pred_heights = decoded[:, [1, 4]]
        # Half-width in physical units
        pred_scales = decoded[:, [2, 5]]
        pred_widths = np.sqrt(2.0 * np.log(2.0)) * pred_scales
    else:
        pred_locs = decoded[:, :k_peaks]
        pred_heights = decoded[:, k_peaks : 2 * k_peaks] if decoded.shape[1] >= 2 * k_peaks else decoded[:, :k_peaks]
    # Ensure consistent peak ordering: sort by predicted location
    if sort_peaks and k_peaks == 2 and pred_locs.shape[1] == 2:
        for i in range(n_samples):
            if pred_locs[i, 0] > pred_locs[i, 1]:
                pred_locs[i, [0, 1]] = pred_locs[i, [1, 0]]
                pred_heights[i, [0, 1]] = pred_heights[i, [1, 0]]
                if pred_widths is not None:
                    pred_widths[i, [0, 1]] = pred_widths[i, [1, 0]]
                if pred_etas is not None:
                    pred_etas[i, [0, 1]] = pred_etas[i, [1, 0]]
    return {
        "pred_etas": pred_etas,
        "true_etas": true_etas,
        "pred_locs": pred_locs,
        "true_locs": true_locs,
        "pred_heights": pred_heights,
        "true_heights": true_heights,
        "pred_sigmas": pred_widths / np.sqrt(2.0 * np.log(2.0)) if pred_widths is not None else None,
        "true_sigmas": true_widths / np.sqrt(2.0 * np.log(2.0)) if true_widths is not None else None,
        "pred_widths": pred_widths,
        "true_widths": true_widths,
        "k_peaks": k_peaks,
    }


def select_hardest_signals(raw_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Select ceil(10% of N) spectra with the closest ground-truth peaks.

    For more than two peaks, use the minimum separation of adjacent sorted

    locations. Stable sorting resolves equal separations in loader order.

    """
    if raw_targets.ndim == 3 and raw_targets.shape[-1] >= 4:
        locations = raw_targets[:, :, 2]
    elif raw_targets.ndim == 3 and raw_targets.shape[-1] == 2:
        locations = raw_targets[:, :, 0]
    elif raw_targets.ndim == 2 and raw_targets.shape[1] >= 4:
        locations = raw_targets[:, [0, 2]]
    else:
        raise ValueError("Hardest-signal selection requires at least two peak locations.")
    if len(locations) == 0 or locations.shape[1] < 2:
        raise ValueError("Hardest-signal selection requires nonempty spectra with at least two peaks.")
    if not np.isfinite(locations).all():
        raise ValueError("Ground-truth peak locations must be finite.")
    separations = np.min(np.diff(np.sort(locations, axis=1), axis=1), axis=1)
    n_selected = max(1, (len(locations) + 9) // 10)
    indices = np.argsort(separations, kind="stable")[:n_selected]
    return indices, separations[indices]


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAE, Pearson r, R^2 (coefficient of determination), and Spearman rho."""
    y_true_flat = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_flat = np.asarray(y_pred, dtype=np.float64).ravel()
    mae = float(np.mean(np.abs(y_true_flat - y_pred_flat)))
    var_true = float(np.var(y_true_flat))
    var_pred = float(np.var(y_pred_flat))
    # Pearson correlation coefficient (r)
    if var_true > 1e-12 and var_pred > 1e-12:
        r_val, _ = scipy.stats.pearsonr(y_true_flat, y_pred_flat)
        pearson_r = float(r_val)
    else:
        pearson_r = 0.0
    # Coefficient of determination (R^2)
    ss_tot = float(np.sum((y_true_flat - np.mean(y_true_flat)) ** 2))
    ss_res = float(np.sum((y_true_flat - y_pred_flat) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    # Spearman rank correlation (rho)
    if var_true > 1e-12 and var_pred > 1e-12:
        rho_val, _ = scipy.stats.spearmanr(y_true_flat, y_pred_flat)
        spearman_rho = float(rho_val)
    else:
        spearman_rho = 0.0
    return {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((y_true_flat - y_pred_flat) ** 2))),
        "pearson_r": pearson_r,
        "r2": r2,
        "spearman_rho": spearman_rho,
    }


def compute_all_verification_metrics(unpacked: dict[str, Any]) -> dict[str, Any]:
    """Compute verification metrics overall and per peak for location, height, and width."""
    metrics: dict[str, Any] = {}
    # 1. Peak Location
    metrics["location_overall"] = compute_regression_metrics(
        unpacked["true_locs"], unpacked["pred_locs"]
    )
    k_peaks = unpacked["k_peaks"]
    for k in range(k_peaks):
        metrics[f"location_peak_{k+1}"] = compute_regression_metrics(
            unpacked["true_locs"][:, k], unpacked["pred_locs"][:, k]
        )
    # 2. Peak Height
    metrics["height_overall"] = compute_regression_metrics(
        unpacked["true_heights"], unpacked["pred_heights"]
    )
    for k in range(k_peaks):
        metrics[f"height_peak_{k+1}"] = compute_regression_metrics(
            unpacked["true_heights"][:, k], unpacked["pred_heights"][:, k]
        )
    # 3. Peak Half-Width (if available)
    if unpacked["pred_widths"] is not None and unpacked["true_widths"] is not None:
        metrics["width_overall"] = compute_regression_metrics(
            unpacked["true_widths"], unpacked["pred_widths"]
        )
        for k in range(k_peaks):
            metrics[f"width_peak_{k+1}"] = compute_regression_metrics(
                unpacked["true_widths"][:, k], unpacked["pred_widths"][:, k]
            )
        metrics["has_width"] = True
    else:
        metrics["has_width"] = False
    metrics["has_sigma"] = (unpacked.get("true_sigmas") is not None
                            and unpacked.get("pred_sigmas") is not None)
    if metrics["has_sigma"]:
        metrics["sigma_overall"] = compute_regression_metrics(
            unpacked["true_sigmas"], unpacked["pred_sigmas"])
        for k in range(k_peaks):
            metrics[f"sigma_peak_{k+1}"] = compute_regression_metrics(
                unpacked["true_sigmas"][:, k], unpacked["pred_sigmas"][:, k])
    metrics["has_eta"] = unpacked.get("pred_etas") is not None and unpacked.get("true_etas") is not None
    if metrics["has_eta"]:
        metrics["eta_overall"] = compute_regression_metrics(unpacked["true_etas"], unpacked["pred_etas"])
        for k in range(k_peaks):
            metrics[f"eta_peak_{k+1}"] = compute_regression_metrics(unpacked["true_etas"][:, k], unpacked["pred_etas"][:, k])
    return metrics


def _fig_to_base64(fig: plt.Figure) -> str:
    """Save matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def generate_scatter_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    unit: str,
    metrics: dict[str, float],
    save_path: Path | None = None,
) -> tuple[str, plt.Figure]:
    """Create Expected vs. Predicted scatter plot with identity line and correlation metrics."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    k_peaks = y_true.shape[1] if y_true.ndim == 2 else 1
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    all_true = y_true.ravel()
    all_pred = y_pred.ravel()
    min_val = min(float(np.min(all_true)), float(np.min(all_pred)))
    max_val = max(float(np.max(all_true)), float(np.max(all_pred)))
    padding = (max_val - min_val) * 0.05 if max_val > min_val else 1.0
    # Identity reference line
    ax.plot(
        [min_val - padding, max_val + padding],
        [min_val - padding, max_val + padding],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Identity (y = x)",
        zorder=1,
    )
    if k_peaks > 1 and y_true.ndim == 2:
        for k in range(k_peaks):
            name = f"Peak {k+1}"
            ax.scatter(
                y_true[:, k],
                y_pred[:, k],
                color=colors[k % len(colors)],
                alpha=0.45,
                edgecolors="none",
                s=20,
                label=name,
                zorder=2,
            )
    else:
        ax.scatter(
            all_true,
            all_pred,
            color="#1f77b4",
            alpha=0.45,
            edgecolors="none",
            s=20,
            label="Samples",
            zorder=2,
        )
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    ax.set_xlabel(f"Expected {label}", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Predicted {label}", fontsize=11, fontweight="bold")
    ax.set_title(f"Expected vs. Predicted {label}", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    # Inset metrics box
    metrics_text = (
        f"MAE: {metrics['mae']:.4f}\n"
        f"Pearson r: {metrics['pearson_r']:.4f}\n"
        f"R²: {metrics['r2']:.4f}\n"
        f"Spearman ρ: {metrics['spearman_rho']:.4f}"
    )
    ax.text(
        0.95,
        0.05,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
    )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    b64 = _fig_to_base64(fig)
    return b64, fig


def generate_loss_curve_plot(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> str:
    """Plot training and validation loss per epoch."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = history.get("epochs", list(range(1, max(len(train_loss), len(val_loss)) + 1)))
    if train_loss:
        ax.plot(epochs[:len(train_loss)], train_loss, label="Train Loss", color="#1f77b4", marker="o", markersize=4)
    if val_loss:
        ax.plot(epochs[:len(val_loss)], val_loss, label="Val Loss", color="#ff7f0e", marker="s", markersize=4)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_title("Training and Validation Loss per Epoch", fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return _fig_to_base64(fig)


def generate_test_examples_plots(
    signals: np.ndarray,
    unpacked: dict[str, Any],
    x_axis: np.ndarray,
    n_examples: int = 10,
    save_dir: Path | None = None,
) -> list[str]:
    """Generate plots for n_examples test set spectra with ground truth vs. predicted peaks."""
    b64_list = []
    n_to_plot = min(n_examples, signals.shape[0])
    for i in range(n_to_plot):
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sig = signals[i]
        t_locs = unpacked["true_locs"][i]
        t_heights = unpacked["true_heights"][i]
        p_locs = unpacked["pred_locs"][i]
        p_heights = unpacked["pred_heights"][i]
        ax.plot(x_axis, sig, color="#2b2d42", linewidth=1.5, label="Spectrum Signal")
        # Ground truth peaks
        k_peaks = len(t_locs)
        for k in range(k_peaks):
            ax.axvline(t_locs[k], color="#2ca02c", linestyle="--", alpha=0.8, label="Expected Peak" if k == 0 else "")
            ax.plot([t_locs[k]], [t_heights[k]], marker="*", color="#2ca02c", markersize=10, zorder=4)
        # Predicted peaks
        for k in range(len(p_locs)):
            ax.axvline(p_locs[k], color="#d62728", linestyle=":", alpha=0.8, label="Predicted Peak" if k == 0 else "")
            ax.plot([p_locs[k]], [p_heights[k]], marker="x", color="#d62728", markersize=8, markeredgewidth=2, zorder=4)
        loc_err = np.mean(np.abs(t_locs - p_locs))
        h_err = np.mean(np.abs(t_heights - p_heights))
        ax.set_xlabel("Sample", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_title(
            f"Test Sample #{i+1} | Loc MAE: {loc_err:.2f} | Height MAE: {h_err:.3f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"example_{i+1:02d}.png", dpi=130, bbox_inches="tight")
        b64_list.append(_fig_to_base64(fig))
    return b64_list


def build_html_report(
    train_setup: dict[str, Any],
    network_info: dict[str, Any],
    git_info: dict[str, str],
    metrics: dict[str, Any],
    loss_b64: str | None,
    scatter_height_b64: str,
    scatter_loc_b64: str,
    scatter_width_b64: str | None,
    examples_b64: list[str],
    scatter_sigma_b64: str | None = None,
    scatter_eta_b64: str | None = None,
) -> str:
    """Build a complete, standalone, self-contained HTML report with embedded styles and figures."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    selection = metrics.get("selection")
    selection_html = ""
    if selection is not None:
        selection_html = (
            "<p><strong>Additional analysis: closest-peak 10%</strong><br>"
            f"{selection['n_selected']} of {selection['n_total']} spectra (rounded up). "
            "Ranked by minimum ground-truth peak separation; "
            f"maximum selected separation: {selection['max_selected_separation']:.4f}.</p>"
        )
    # Format metrics table rows
    def fmt(val: float | None) -> str:
        return f"{val:.4f}" if val is not None and np.isfinite(val) else "N/A"
    metric_rows = ""
    dimensions = [("location", "Peak Location (μ)", ""),
                  ("height", "Peak Height (A)", ""),
                  ("sigma", "Peak Width (σ)", ""),
                  ("width", "Peak Half-Width (HWHM)", ""),
                  ("eta", "Peak Shape (η)", "")]
    for prefix, label, unit in dimensions:
        keys = [f"{prefix}_overall"] + [key for key in metrics if key.startswith(f"{prefix}_peak_")]
        if keys[0] not in metrics:
            continue
        for key in keys:
            values = metrics[key]
            overall = key.endswith("overall")
            name = label if overall else "↳ Peak " + key.rsplit("_", 1)[1]
            cls = "" if overall else ' class="sub-row"'
            metric_rows += f"<tr{cls}><td>{name}</td>"
            for field in ("mae", "rmse", "pearson_r", "r2", "spearman_rho"):
                suffix = ""
                metric_rows += f"<td>{fmt(values.get(field))}{suffix}</td>"
            metric_rows += "</tr>"
    sigma_scatter_html = (f'<div class="plot-box"><img src="{scatter_sigma_b64}" alt="Sigma Scatter Plot" /></div>' if scatter_sigma_b64 else "")
    eta_scatter_html = (f'<div class="plot-box"><img src="{scatter_eta_b64}" alt="Eta Scatter Plot" /></div>' if scatter_eta_b64 else "")
    generalized_summary_html = ""
    if "per_target" in metrics:
        generalized_summary_html = (
            f"<p><strong>Mean R²:</strong> {fmt(metrics['mean_r2'])}; <strong>n_test:</strong> {metrics['n_test']}</p>")
    setup_extra_html = "".join(
        f'<div class="key-value"><span class="key">{escape(str(key))}</span><span class="val">{escape(str(value))}</span></div>'
        for key, value in train_setup.items()
        if key not in {"epochs", "batch_size", "lr", "generator_type", "n_train", "n_val", "n_test", "distinct_epochs", "train_seed", "val_seed", "test_seed"})
    # Examples HTML
    examples_html = ""
    for idx, b64 in enumerate(examples_b64, 1):
        examples_html += f"""
        <div class="example-card">
            <h4>Example {idx}</h4>
            <img src="{b64}" alt="Test Example {idx}" />
        </div>
        """
    loss_section = ""
    if loss_b64:
        loss_section = f"""
        <section class="card">
            <h2>5. Training & Validation Loss Curve</h2>
            <div class="center-img">
                <img src="{loss_b64}" alt="Loss per Epoch" style="max-width: 700px; width: 100%;" />
            </div>
        </section>
        """
    width_scatter_html = ""
    if scatter_width_b64:
        width_scatter_html = f"""
        <div class="plot-box">
            <img src="{scatter_width_b64}" alt="Half-Width Scatter Plot" />
        </div>
        """
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Training & Verification Report - {network_info.get('network_name', 'Spectrum Model')}</title>
<style>
    :root {{
        --primary: #0f4c81;
        --secondary: #2b2d42;
        --accent: #e63946;
        --bg: #f8f9fa;
        --card-bg: #ffffff;
        --border: #e9ecef;
        --text: #212529;
        --muted: #6c757d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: var(--bg);
        color: var(--text);
        line-height: 1.6;
        margin: 0;
        padding: 24px;
    }}
    .container {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    header {{
        background: var(--card-bg);
        padding: 24px 32px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        border-left: 6px solid var(--primary);
    }}
    h1 {{ margin: 0; font-size: 24px; color: var(--primary); }}
    .meta-tag {{ font-size: 13px; color: var(--muted); }}
    .badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background: #e3f2fd;
        color: var(--primary);
    }}
    .card {{
        background: var(--card-bg);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        border: 1px solid var(--border);
    }}
    h2 {{
        margin-top: 0;
        font-size: 18px;
        border-bottom: 2px solid var(--border);
        padding-bottom: 8px;
        color: var(--secondary);
    }}
    .grid-2 {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 20px;
    }}
    .grid-plots {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
        gap: 20px;
        justify-items: center;
    }}
    .grid-examples {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
        gap: 20px;
    }}
    .example-card {{
        background: #fdfdfd;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }}
    .example-card h4 {{ margin: 0 0 8px 0; font-size: 14px; color: var(--muted); }}
    .example-card img {{ width: 100%; border-radius: 4px; }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 12px;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid var(--border);
        font-size: 14px;
    }}
    th {{
        background-color: #f1f3f5;
        color: var(--secondary);
        font-weight: 600;
    }}
    .sub-row td {{
        padding-left: 28px;
        color: #495057;
        font-size: 13.5px;
    }}
    .key-value {{
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dotted var(--border);
        font-size: 14px;
    }}
    .key-value .key {{ font-weight: 600; color: #495057; }}
    .key-value .val {{ font-family: monospace; color: #212529; }}
    .center-img {{ text-align: center; margin: 12px 0; }}
    .plot-box img {{ max-width: 100%; height: auto; border-radius: 6px; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <div>
            <h1>Training & Verification Report</h1>
            <div class="meta-tag">Generated on {timestamp}</div>
        </div>
        <div>
            <span class="badge">{network_info.get('network_name', 'Unknown Network')}</span>
            <span class="badge">{network_info.get('preprocessing_name', 'Unknown Preprocessing')}</span>
        </div>
    </header>
    <div class="grid-2">
        <section class="card">
            <h2>1. Training Setup & Hyperparameters</h2>
            <div class="key-value"><span class="key">Epochs</span><span class="val">{train_setup.get('epochs', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Batch Size</span><span class="val">{train_setup.get('batch_size', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Learning Rate</span><span class="val">{train_setup.get('lr', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Generator Type</span><span class="val">{train_setup.get('generator_type', 'gaussian')}</span></div>
            <div class="key-value"><span class="key">Training Samples</span><span class="val">{train_setup.get('n_train', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Validation Samples</span><span class="val">{train_setup.get('n_val', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Test Samples</span><span class="val">{train_setup.get('n_test', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Distinct Epochs</span><span class="val">{train_setup.get('distinct_epochs', False)}</span></div>
            <div class="key-value"><span class="key">Seeds (Train / Val / Test)</span><span class="val">{train_setup.get('train_seed', 'N/A')} / {train_setup.get('val_seed', 'N/A')} / {train_setup.get('test_seed', 'N/A')}</span></div>
            {setup_extra_html}
        </section>
        <section class="card">
            <h2>2. Network, Preprocessing & Git</h2>
            <div class="key-value"><span class="key">Network Architecture</span><span class="val">{network_info.get('network_name', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Preprocessing Transform</span><span class="val">{network_info.get('preprocessing_name', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Preprocessing Channels</span><span class="val">{network_info.get('in_channels', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Trainable Parameters</span><span class="val">{network_info.get('params', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Git Hash</span><span class="val">{git_info.get('githash', 'N/A')[:12]}</span></div>
            <div class="key-value"><span class="key">Git Branch</span><span class="val">{git_info.get('gitbranch', 'N/A')}</span></div>
            <div class="key-value"><span class="key">Git Path / Remote</span><span class="val">{git_info.get('gitpath', 'N/A')}</span></div>
        </section>
    </div>
    {loss_section}
    <section class="card">
        <h2>6. Verification Metrics Summary</h2>
        {selection_html}
        {generalized_summary_html}
        <table>
            <thead>
                <tr>
                    <th>Target Dimension</th>
                    <th>Mean Absolute Error (MAE)</th>
                    <th>Root Mean Square Error (RMSE)</th>
                    <th>Pearson Correlation (r)</th>
                    <th>Coeff. of Determination (R²)</th>
                    <th>Spearman Rank (ρ)</th>
                </tr>
            </thead>
            <tbody>
                {metric_rows}
            </tbody>
        </table>
    </section>
    <section class="card">
        <h2>7. Expected vs. Predicted Scatter Plots</h2>
        <div class="grid-plots">
            <div class="plot-box">
                <img src="{scatter_loc_b64}" alt="Location Scatter Plot" />
            </div>
            <div class="plot-box">
                <img src="{scatter_height_b64}" alt="Height Scatter Plot" />
            </div>
            {width_scatter_html}
            {sigma_scatter_html}
            {eta_scatter_html}
            <p>For Gaussian peaks: HWHM = √(2 ln 2) × σ.</p>
        </div>
    </section>
    <section class="card">
        <h2>8. Test Set Spectrum Prediction Examples ({len(examples_b64)} Samples)</h2>
        <div class="grid-examples">
            {examples_html}
        </div>
    </section>
</div>
</body>
</html>
"""
    return html


def run_verification(
    model: BaseSpectrumNetwork,
    test_loader: DataLoader,
    output_dir: str | Path,
    history: dict[str, list[float]] | None = None,
    train_setup: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
    n_examples: int = 10,
    n_outputs: int | None = None,
) -> dict[str, Any]:
    """Verify the full test set plus its closest-peak 10% using one inference pass.

    One report.html contains both populations; general-result plots are saved in plots/.

    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.to(device)
    if n_outputs is not None:
        if n_outputs not in (6, 8):
            raise ValueError("n_outputs must be 6 or 8")
        from cwt_processing.synthetic_peak_generator import flatten_targets as flatten_physical_targets, output_names
    flat_targets = []
    all_decoded: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    signal_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch_idx, (signals, raw_targets) in enumerate(test_loader):
            signals_dev = signals.to(device)
            preds, context = model(signals_dev)
            decoded = model.decode_predictions(preds, context)
            all_decoded.append(decoded.detach().cpu().numpy())
            if n_outputs is not None:
                truth = flatten_physical_targets(raw_targets.to(device), n_outputs).detach().cpu().numpy()
                prediction = all_decoded[-1]
                if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != n_outputs:
                    raise ValueError("Expected matching (N, n_outputs) physical targets and predictions")
                flat_targets.append(truth)
                peaks = truth.reshape(len(truth), 2, n_outputs // 2)
                physical = np.zeros((len(truth), 2, 5))
                physical[:, :, 1] = peaks[:, :, 1]
                physical[:, :, 2] = peaks[:, :, 0]
                physical[:, :, 3] = peaks[:, :, 2]
                if n_outputs == 8:
                    physical[:, :, 4] = peaks[:, :, 3]
                all_targets.append(physical)
            else:
                all_targets.append(raw_targets.detach().cpu().numpy())
            signal_batches.append(signals.detach().cpu().numpy())
    if not all_decoded:
        raise ValueError("test_loader is empty")
    decoded_arr = np.concatenate(all_decoded, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    signals_arr = np.concatenate(signal_batches, axis=0)
    del signal_batches
    # Unpack targets and predictions
    unpacked = unpack_predictions_and_targets(decoded_arr, targets_arr, sort_peaks=n_outputs is None)
    # Compute regression & correlation metrics
    metrics = compute_all_verification_metrics(unpacked)
    # Reuse the same predictions and labels to compare the closest-peak subset.
    indices, separations = select_hardest_signals(targets_arr)
    hardest_unpacked = {
        key: value[indices] if isinstance(value, np.ndarray) else value
        for key, value in unpacked.items()
    }
    hardest_metrics = compute_all_verification_metrics(hardest_unpacked)
    hardest_metrics["selection"] = {
        "fraction": 0.10,
        "criterion": "minimum ground-truth peak separation",
        "n_total": len(targets_arr),
        "n_selected": len(indices),
        "max_selected_separation": float(separations[-1]),
        "selected_indices": indices.tolist(),
    }
    if history is None and (output_path / "metrics.csv").exists():
        history = load_loss_history_from_csv(output_path / "metrics.csv")
    if n_outputs is not None:
        y_true = np.concatenate(flat_targets)
        names = list(output_names(n_outputs))
        metrics.update(_generalized_summary(y_true, decoded_arr, names, n_outputs))
        hardest_metrics.update(_generalized_summary(y_true[indices], decoded_arr[indices], names, n_outputs))
        np.savez_compressed(output_path / "test_predictions.npz", y_true=y_true,
                            y_pred=decoded_arr, signals=signals_arr, output_names=np.asarray(names))
    full_html = _render_verification_population(
        model, test_loader, output_path, unpacked, signals_arr,
        metrics, history, train_setup, n_examples, save_plots=True,
    )
    subset_html = _render_verification_population(
        model, test_loader, output_path, hardest_unpacked, signals_arr[indices],
        hardest_metrics, {}, train_setup, n_examples,
    )
    marker = '    <section class="card">\n        <h2>6. Verification Metrics Summary</h2>'
    subset_sections = subset_html[subset_html.index(marker):subset_html.rindex("</div>")]
    full_html = full_html.replace(marker, '<p id="all-samples" class="badge">All test spectra</p>\n' + marker, 1)
    insertion = full_html.rindex("</div>")
    metadata = json.dumps(_json_safe({"all_samples": metrics, "hardest_10_percent": hardest_metrics}), allow_nan=False).replace("<", "\\\u003c")
    full_html = (full_html[:insertion]
        + '<p id="hardest-10-percent" class="badge">Closest-peak 10% of test spectra</p>\n'
        + subset_sections
        + '<script id="verification-metrics" type="application/json">' + metadata + '</script>\n'
        + full_html[insertion:])
    # Number all sections consecutively after joining both analyses.
    section_number = 0
    def renumber_section(match: re.Match) -> str:
        nonlocal section_number
        section_number += 1
        return f"<h2>{section_number}. {match.group(1)}</h2>"
    full_html = re.sub(r"<h2>\d+\.\s*(.*?)</h2>", renumber_section, full_html)
    report_path = output_path / "report.html"
    report_path.write_text(full_html, encoding="utf-8")
    metrics_path = output_path / "metrics.json"
    saved_metrics = dict(metrics)
    saved_metrics["hardest_10_percent"] = hardest_metrics
    metrics_path.write_text(json.dumps(_json_safe(saved_metrics), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {"metrics": metrics, "report_path": report_path, "metrics_path": metrics_path, "plots_dir": output_path / "plots",
            "hardest_10_percent": {"metrics": hardest_metrics, "report_path": report_path}}


def _render_verification_population(
    model: BaseSpectrumNetwork,
    test_loader: DataLoader,
    output_path: Path,
    unpacked: dict[str, Any],
    signals_arr: np.ndarray,
    metrics: dict[str, Any],
    history: dict[str, list[float]] | None,
    train_setup: dict[str, Any] | None,
    n_examples: int,
    save_plots: bool = False,
) -> str:
    """Render a population; optionally save its plots alongside the report."""
    plots_dir = output_path / "plots"
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    # Git metadata
    git_info = get_git_info()
    # Network metadata
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prep_channels = getattr(getattr(model, "preprocessing", None), "out_channels", "N/A")
    hparams = getattr(model, "hparams", {})
    network_info = {
        "network_name": hparams.get("network_name", model.__class__.__name__),
        "preprocessing_name": hparams.get("preprocessing_name", getattr(getattr(model, "preprocessing", None), "__class__", type(None)).__name__),
        "in_channels": prep_channels,
        "params": f"{n_params:,}",
    }
    if train_setup is None:
        train_setup = {
            "epochs": hparams.get("epochs", "N/A"),
            "batch_size": hparams.get("batch_size", "N/A"),
            "lr": getattr(model, "lr", "N/A"),
            "generator_type": hparams.get("generator_type", "gaussian"),
            "n_train": hparams.get("n_train", "N/A"),
            "n_val": hparams.get("n_val", "N/A"),
            "n_test": len(unpacked["true_locs"]),
            "distinct_epochs": hparams.get("distinct_epochs", False),
            "train_seed": hparams.get("train_seed", "N/A"),
            "val_seed": hparams.get("val_seed", "N/A"),
            "test_seed": hparams.get("test_seed", "N/A"),
        }
    # Generate plots
    scatter_loc_b64, _ = generate_scatter_plot(
        unpacked["true_locs"],
        unpacked["pred_locs"],
        label="Peak Location",
        unit="",
        metrics=metrics["location_overall"],
        save_path=plots_dir / "scatter_location.png" if save_plots else None,
    )
    scatter_height_b64, _ = generate_scatter_plot(
        unpacked["true_heights"],
        unpacked["pred_heights"],
        label="Peak Height",
        unit="",
        metrics=metrics["height_overall"],
        save_path=plots_dir / "scatter_height.png" if save_plots else None,
    )
    scatter_width_b64 = None
    if metrics["has_width"] and unpacked["pred_widths"] is not None:
        scatter_width_b64, _ = generate_scatter_plot(
            unpacked["true_widths"],
            unpacked["pred_widths"],
            label="Peak Half-Width",
            unit="",
            metrics=metrics["width_overall"],
            save_path=plots_dir / "scatter_width.png" if save_plots else None,
        )
    scatter_sigma_b64 = None
    if metrics.get("has_sigma", False):
        scatter_sigma_b64, _ = generate_scatter_plot(
            unpacked["true_sigmas"], unpacked["pred_sigmas"],
            label="Peak Width (σ)", unit="", metrics=metrics["sigma_overall"],
            save_path=plots_dir / "scatter_sigma.png" if save_plots else None,
        )
    scatter_eta_b64 = None
    if metrics.get("has_eta", False):
        scatter_eta_b64, _ = generate_scatter_plot(
            unpacked["true_etas"], unpacked["pred_etas"], label="Peak Shape (η)",
            unit="", metrics=metrics["eta_overall"],
            save_path=plots_dir / "scatter_eta.png" if save_plots else None)
    if save_plots and "output_names" in metrics:
        group_arrays = {"mu": ("true_locs", "pred_locs"), "A": ("true_heights", "pred_heights"),
                        "sigma": ("true_sigmas", "pred_sigmas"), "eta": ("true_etas", "pred_etas")}
        for i, name in enumerate(metrics["output_names"]):
            stride = len(metrics["output_names"]) // 2
            group = ["mu", "A", "sigma", "eta"][i % stride]
            true_key, pred_key = group_arrays[group]
            peak = i // stride
            generate_scatter_plot(unpacked[true_key][:, peak], unpacked[pred_key][:, peak],
                                  label=str(name), unit="", metrics=compute_regression_metrics(
                                      unpacked[true_key][:, peak], unpacked[pred_key][:, peak]),
                                  save_path=plots_dir / f"scatter_{name}.png")
    loss_b64 = None
    if history is None and (output_path / "metrics.csv").exists():
        history = load_loss_history_from_csv(output_path / "metrics.csv")
    if history and ("train_loss" in history or "val_loss" in history):
        loss_b64 = generate_loss_curve_plot(history, save_path=plots_dir / "loss_curve.png" if save_plots else None)
    # Generate example predictions
    signals_arr = signals_arr[:max(0, n_examples)]
    ds = getattr(test_loader, "dataset", None)
    if ds is not None and hasattr(ds, "_x"):
        x_axis = np.asarray(ds._x)
    else:
        x_axis = np.arange(signals_arr.shape[1])
    examples_b64 = generate_test_examples_plots(
        signals=signals_arr,
        unpacked={
            "true_locs": unpacked["true_locs"][: len(signals_arr)],
            "true_heights": unpacked["true_heights"][: len(signals_arr)],
            "pred_locs": unpacked["pred_locs"][: len(signals_arr)],
            "pred_heights": unpacked["pred_heights"][: len(signals_arr)],
        },
        x_axis=x_axis,
        n_examples=n_examples,
        save_dir=plots_dir / "examples" if save_plots else None,
    )
    # Assemble HTML Report
    html_content = build_html_report(
        train_setup=train_setup,
        network_info=network_info,
        git_info=git_info,
        metrics=metrics,
        loss_b64=loss_b64,
        scatter_height_b64=scatter_height_b64,
        scatter_loc_b64=scatter_loc_b64,
        scatter_width_b64=scatter_width_b64,
        examples_b64=examples_b64,
        scatter_sigma_b64=scatter_sigma_b64,
        scatter_eta_b64=scatter_eta_b64,
    )
    return html_content


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metric(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = compute_regression_metrics(y_true, y_pred)
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return {"R2": m["r2"] if np.var(yt) > 1e-12 else float("nan"),
            "MAE": m["mae"], "RMSE": m["rmse"],
            "Pearson_r": m["pearson_r"] if np.var(yt) > 1e-12 and np.var(yp) > 1e-12 else float("nan"),
            "Spearman_rho": m["spearman_rho"] if np.var(yt) > 1e-12 and np.var(yp) > 1e-12 else float("nan")}


def _generalized_summary(y_true, y_pred, names, n_outputs):
    groups = {"mu": [0, 3], "A": [1, 4], "sigma": [2, 5]} if n_outputs == 6 else {
        "mu": [0, 4], "A": [1, 5], "sigma": [2, 6], "eta": [3, 7]}
    per_target = {name: _metric(y_true[:, i], y_pred[:, i]) for i, name in enumerate(names)}
    finite = [m["R2"] for m in per_target.values() if np.isfinite(m["R2"])]
    return {"mean_r2": float(np.mean(finite)) if finite else float("nan"),
            "per_target": per_target,
            "groups": {key: _metric(y_true[:, idx], y_pred[:, idx]) for key, idx in groups.items()},
            "n_test": len(y_true), "output_names": names}


@torch.no_grad()
def collect_predictions(model, test_loader, *, n_outputs):
    from cwt_processing.synthetic_peak_generator import flatten_targets as flatten_physical_targets
    if n_outputs not in (6, 8):
        raise ValueError("n_outputs must be 6 or 8")
    device = next(model.parameters()).device
    model.eval()
    truths, predictions, signals_all = [], [], []
    for signals, targets in test_loader:
        pred, context = model(signals.to(device))
        predictions.append(model.decode_predictions(pred, context).detach().cpu().numpy())
        truths.append(flatten_physical_targets(targets.to(device), n_outputs).detach().cpu().numpy())
        signals_all.append(signals.detach().cpu().numpy())
    if not truths:
        raise ValueError("test_loader is empty")
    return np.concatenate(truths), np.concatenate(predictions), np.concatenate(signals_all)


def run_generalized_reporting(*, model, test_loader, output_dir, train_setup, n_outputs):
    result = run_verification(model, test_loader, output_dir, train_setup=train_setup,
                              device=next(model.parameters()).device, n_outputs=n_outputs)
    summary = {key: result["metrics"][key] for key in
               ("mean_r2", "per_target", "groups", "n_test", "output_names")}
    summary["report_path"] = str(result["report_path"])
    summary["hardest_10_percent"] = result["hardest_10_percent"]["metrics"]
    return summary
