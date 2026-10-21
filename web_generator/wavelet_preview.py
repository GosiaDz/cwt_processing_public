"""Existing web-only CWT preview kernels, not used for model training."""
import numpy as np
import math
WAVELET_TYPES=("GD1","MH / GD2","GD3")
def wavelet_kernel(kind: str, scale: float) -> np.ndarray:
    """Return a zero-mean, L2-normalized discrete Gaussian-derivative wavelet.

    ``scale`` is expressed in sample points.  The supported preview wavelets are
    the first Gaussian derivative (GD1), the Mexican Hat / second derivative
    (MH / GD2), and the third Gaussian derivative (GD3).
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Wavelet scale must be a finite number > 0.")
    if kind not in WAVELET_TYPES:
        raise ValueError(f"Unsupported wavelet: {kind}")

    radius = max(1, int(math.ceil(4.0 * scale)))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    u = t / scale
    g = np.exp(-0.5 * u * u)

    if kind == "GD1":
        k = -u * g
    elif kind == "MH / GD2":
        # Conventional Mexican Hat sign; the opposite sign is equivalent
        # for feature localization.
        k = (1.0 - u * u) * g
    else:  # GD3
        k = (3.0 * u - u ** 3) * g

    # Discrete truncation can leave a tiny DC component.
    k = k - np.mean(k)
    norm = float(np.linalg.norm(k))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Wavelet kernel collapsed at this scale.")
    return k / norm


def wavelet_transform_1d(signal: np.ndarray, kind: str, scale: float = 1.0) -> np.ndarray:
    """Convolve a 1-D signal with a preview wavelet using edge replication.

    The returned array has exactly the same length as ``signal``.
    """
    y = np.asarray(signal, dtype=np.float64)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("Wavelet preview expects a one-dimensional signal.")
    if not np.all(np.isfinite(y)):
        raise ValueError("Wavelet preview signal contains non-finite values.")

    k = wavelet_kernel(kind, scale)
    radius = (len(k) - 1) // 2
    padded = np.pad(y, (radius, radius), mode="edge")
    # Reverse the kernel for mathematical convolution.  For display, the
    # sign convention is not scientifically material, but keeping a true
    # convolution makes the definition explicit.
    out = np.convolve(padded, k, mode="valid")
    return out.astype(np.float64, copy=False)


