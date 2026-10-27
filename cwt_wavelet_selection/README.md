# CWT wavelet and scale selection

`cwt_wavelet_scale_selection.py` compares CWT wavelets and selects scales for resolving two overlapping peaks. It supports Gaussian, Lorentzian, and pseudo-Voigt peak families. The tuning and independent test sets use separate fixed seeds; results are evaluated at noise levels from 0% to 5%.

## Requirements

Run from this directory with Python and the project dependencies installed: NumPy, SciPy, PyWavelets, and the generator's PyTorch dependency. Keep `generalized_peak_datasets.py` alongside the script.

## Usage

```bash
python cwt_wavelet_scale_selection.py --family gaussian
python cwt_wavelet_scale_selection.py --family lorentzian --n-tuning 500 --n-test 1000 --n-null 500
python cwt_wavelet_scale_selection.py --family pseudovoigt --report --csv --output-dir results
```

`--family` is required and accepts `gaussian`, `lorentzian`, or `pseudovoigt`. Sample counts default to 1,000 tuning geometries, 2,000 test geometries, and 1,000 pure-noise realizations. Use `--report` (or `--raport`) to create a self-contained HTML report and `--csv` to create detailed CSV results; both are off by default. Files are named `cwt_selection_<family>_report.html` and `cwt_selection_<family>_results.csv` in the selected output directory.

By default, recommendations are based on tuning-set success, within 5 percentage points of the best wavelet. Add `--background-limit` to require the background-peak signal rate to be at most 5% for a full recommendation. The script prints selected scales and recommendation status; test metrics are saved in the optional report and CSV.

The experiment can take a while at default sample counts. For a quick run, lower all three sample counts; use the defaults for comparable full runs.
