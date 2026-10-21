# Validation performed in preparation

**Source inputs:** `cwt_processing-main(1)(1).zip`, `double-peak-generator-main (2).zip`, `CWT_PROCESSING_TESTY_I_POLECENIA.txt`, and the HTML report from 2026-10-06.

1. Selected `experiment_engine.py` plus the small number of direct dependencies from the original repository, rather than the old unrelated networks.
2. Updated reporter imports to point at the same `synthetic_peak_generator.py` as the training engine.
3. `pytest -q tests` passed: 12 tests, covering bitwise-identical generated spectra and targets for all 5 supported families with and without noise, centered-pair constraint, exported ZIP, and inference model shape.
4. Executed a real 60-sample, 1-epoch Gaussian MH+GD smoke training on CPU. The run produced checkpoints, `report.html`, `metrics.json`, and `resolved_config.json`. Its mean R² was negative due to extremely brief training; this is **not** evidence against the full run.
5. Loaded that small-run checkpoint with `transformer/predict_checkpoint.py` and successfully exported predictions for all 12 supplied example samples.
6. Did not rerun the original 50,000-sample/50-epoch experiment; its reported mean R² = 0.9980 has **not** been independently reproduced.
7. Streamlit browser/cloud deployment has **not** been exercised here. Streamlit is not installed in the working environment and package installation from PyPI was unavailable due to DNS/network restrictions; it should be validated on a new private test deployment before changing the existing public page.

No private experimental spectra, collaborators' access tokens, whole-paper PDFs, historic checkpoints or external dependencies have been added to this archive.
