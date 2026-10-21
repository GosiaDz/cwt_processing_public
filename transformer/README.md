# Transformer — reproducible centered-peak experiments

## Selected implementation

The executable engine is `src/cwt_processing/experiment_engine.py`. It uses:

- `src/cwt_processing/synthetic_peak_generator.py`: single, deterministic synthetic-data generator.
- `src/cwt_processing/networks/backbones.py`, `networks/base.py`: patch Transformer and Lightning wrapper.
- `src/cwt_processing/preprocessing/dual_wavelet.py`, `preprocessing/base.py`: MH/GD convolution kernels; additional wavelets through PTWT.
- `src/cwt_processing/verification.py`: verification metrics and HTML report.
- `predict_checkpoint.py`: restore a trained checkpoint and predict parameters for unseen NPZ spectra.

Old experimental networks and the independent legacy peak generator were **not** copied. The reporter's old generator import was replaced by the native generator's `flatten_targets` and `output_names` helpers. This prevents train/report parameter conventions from drifting apart.

## Full historic Gaussian experiment (Windows PowerShell)

Run from the root of the repository, after `pip install -e .`:

```powershell
.\.venv\Scripts\python.exe -m cwt_processing.experiment_engine --samples 50000 --family gaussian --wavelets "MH;GD" --scales "5;23" --epochs 50 --output-dir outputs\gauss_MH_GD_5_23
```

The engine's defaults are: fixed center, axis 0–400 and 400 points, Δμ≥10, no noise, 40k/5k/5k split, seed 31337/7331/1337, 50 epochs, batch 256, AdamW with learning rate 1e-3, weight decay 1e-4, and ReduceLROnPlateau. CWT has four input channels (MH@5, MH@23, GD@5, GD@23), and the six regression outputs are μ1, A1, σ1, μ2, A2, σ2.

For a low-cost technical check:

```powershell
.\.venv\Scripts\python.exe -m cwt_processing.experiment_engine --samples 60 --epochs 1 --batch-size 16 --wavelets "MH;GD" --scales "5;23" --accelerator cpu --output-dir outputs\smoke
```

**The small run verifies that the software executes; its R² is not an estimate of the 50-epoch model performance.**

## Additional experiments

```powershell
# Lorentzian
.\.venv\Scripts\python.exe -m cwt_processing.experiment_engine --family lorentzian --samples 50000 --wavelets "MH;GD" --scales "5;23"

# Pseudo-Voigt; predicts eight parameters, including eta1 and eta2
.\.venv\Scripts\python.exe -m cwt_processing.experiment_engine --family pseudovoigt --samples 50000 --wavelets "MH;GD" --scales "5;23"

# Gaussian / Lorentzian / pseudo-Voigt independently mixed in one sample
.\.venv\Scripts\python.exe -m cwt_processing.experiment_engine --family independent_mix --samples 50000 --wavelets "MH;GD" --scales "5;23"

# Non-custom CWT wavelets require: pip install -e '.[wavelets]'
.\.venv\Scripts\python.exe -m cwt_processing.experiment_engine --wavelets "MH;GD;gaus1" --scales "5;23;46" --samples 50000
```

The experiment saves `resolved_config.json`, `report.html`, `metrics.json`, `engine_summary.json` and trained checkpoint(s) in `outputs/`.

## Use a saved checkpoint to predict new samples

From the repo root:

```powershell
.\.venv\Scripts\python.exe transformer\predict_checkpoint.py --run "outputs\gauss_MH_GD_5_23\RUN_FOLDER" --input examples\publication_gaussian_seed31337_N12.npz --output outputs\predictions.csv
```

Substitute the **actual** output folder containing `resolved_config.json` and `best-*.ckpt`; the engine may create a timestamped folder. `predict_checkpoint.py` restores architecture, preprocessing, scaling statistics and weights from that run, then exports predictions (CSV). Gaussian/Lorentzian checkpoints output 6 targets; pseudo-Voigt/mixed checkpoints output 8 targets. Prediction is valid only for inputs compatible with the checkpoint's training domain and number of axis points.

To reproduce the example dataset: `python examples/make_example.py`. The exported `summed` spectra are byte-identical to the Tensor returned by `SyntheticPeakDataset` with matching seed/index/config, as asserted in the tests.
