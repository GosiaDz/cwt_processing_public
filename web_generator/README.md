# Online Double Peak Generator (Streamlit)

## Purpose

`app.py` is the interactive browser interface for **the identical** `SyntheticPeakDataset` used by Transformer training. It has no independent mathematical peak sampler. The lightweight `bridge.py` only formats and exports the dataset. `wavelet_preview.py` provides visualization-only filters adapted from the previous application's preview controls.

## Features

- Gaussian, Lorentzian, pseudo-Voigt, paired and independent mixtures.
- Native publication defaults: fixed center 200; x=0..400, 400 points; Gaussian; ≥10 axis units between peaks; 0% noise; random seed 31337.
- Editable geometry, widths, amplitude ratios, pseudo-Voigt eta, random center (optional), and noise (percentage of summed-signal maximum as Gaussian standard deviation).
- Preview of peak 1, peak 2, summed signal and selected Gaussian derivative / Mexican-hat wavelet overlays.
- Dataset ZIP including `spectra.npz` (`summed`, `peak1`, `peak2`, `targets`, `x_axis`), `labels.csv` and `config.json`; separate CSV label download.
- Chunked generation for consistent absolute sample indexing. Online cap 10,000 samples to control server memory; larger experiments should be run from the Transformer CLI.

The default web sample count is 1,000 for responsiveness, not the 50,000 used in the supplied training report. A small downloaded sample bank is in `examples/`.

## Run locally

```bash
pip install -e '.[web,wavelets]'
streamlit run web_generator/app.py
```

## Deploy safely to Streamlit Community Cloud

**Do not edit the old, working Double Peak Generator repository.** First upload this new repository to GitHub and deploy it as a **separate test app**.

1. Log in to the Streamlit dashboard and create a new app using the new `cwt_processing_public` GitHub repository.
2. Select the appropriate branch (usually `main`) and **Main file path: `web_generator/app.py`**.
3. Confirm that the root `requirements.txt` is used. It installs the shared package so `SyntheticPeakDataset` is available to the app. Choose Python 3.12 if prompted.
4. Generate data with defaults; download a ZIP; compare the samples with `examples/publication_gaussian_seed31337_N12.npz` when the sample count and seed match.
5. Only after the replacement is verified decide whether to retire the old deployment and reuse its preferred app name. Keeping the **exact old web address is not guaranteed**: Streamlit may require redeployment and name availability. Take note of the old URL and configuration first.

### Compatibility note

The prior standalone app included an independent-range/skew-normal and a special CA–FA legacy mode, with its own mathematical conventions. They are intentionally **not part of this scientific reproduction package**: silently retaining these modes inside the shared generator would violate the requirement of numerical identity with Transformer data. The scientific families G/L/PV and their mixtures, interactive plots, wavelet preview, and NPZ/CSV/ZIP exports are included. If older specialty functionality is needed in the future, it should remain clearly isolated and documented rather than claiming equivalence to the publication generator.
