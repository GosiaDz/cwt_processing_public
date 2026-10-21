# CWT processing — public research code

**Overlapping synthetic peaks · shared centered generator · multi-wavelet Transformer**

This repository is a minimal public package for research reproducibility. It contains **one mathematical peak generator**, `transformer/src/cwt_processing/synthetic_peak_generator.py`, used directly by the CWT Transformer training engine and the interactive Streamlit application.

| Component | Where | Instructions |
| --- | --- | --- |
| Transformer and experiment engine | [`transformer/`](transformer/) | [`transformer/README.md`](transformer/README.md) |
| Online double-peak generator | [`web_generator/`](web_generator/) | [`web_generator/README.md`](web_generator/README.md) |
| Downloadable reference samples | [`examples/`](examples/) | `publication_gaussian_seed31337_N12.npz`, `.csv`, `.json` |
| Identical-data regression tests | [`tests/`](tests/) | `pytest -q` |

## Research baseline

The supplied report dated **2026-10-06** describes a 50,000-sample Gaussian experiment, 40k/5k/5k split, fixed center at 200, 400 axis points from 0 to 400, zero noise, dual MH + GD CWT (scales 5 and 23), 50 epochs, and a Transformer. The historic report reported mean R² = 0.9980. **This is a result from the user-supplied historic report, not a result reproduced by this package's small smoke test.** The original report has no Git commit hash and original trained model weights have not been provided.

Other supported peak families: Lorentzian, pseudo-Voigt, paired Gaussian/Lorentzian/pseudo-Voigt mixtures, and independently mixed families. For all families, the same Gaussian-equivalent FWHM convention is used for σ. No independent second sampler is maintained by Streamlit.

## First run (Python 3.12 recommended)

From the **top level** of this repository:

```bash
python -m venv .venv
# activate the environment using your operating system's instructions
python -m pip install -e '.[web,wavelets,test]'
python -m pytest -q
streamlit run web_generator/app.py
```

Windows PowerShell without activating the venv:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[web,wavelets,test]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run web_generator/app.py
```

For a short training run:

```powershell
.\.venv\Scripts\python.exe -m cwt_processing.experiment_engine --samples 500 --epochs 2 --wavelets "MH;GD" --scales "5;23" --output-dir outputs\quick_test
```

Then see the two component-specific READMEs above.

## Publication/use notes

- Keep original project repositories and deployed Streamlit application untouched until the new test app has been verified.
- This ZIP is a **source-code upload package**, not a trained model distribution; checkpoints are produced during training.
- The Streamlit wavelet overlays are **visual previews** only. The Transformer's training CWT is produced by the actual experiment engine.
- **Before making this repository public**, check the rights of all source-code contributors and decide together on an appropriate license. No license has been presumed on their behalf. See [`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md).
