# Before public release

- [ ] Confirm the research group and all code contributors permit public release of the files selected from the private research repository.
- [ ] Choose and include a source-code LICENSE; none is supplied here without the contributors' decision.
- [ ] Confirm manuscript formulas and ranges match the centered generator in this archive.
- [ ] Run `python -m pytest -q` with the intended deployment version of Python.
- [ ] Run at least one small training experiment and check the HTML report, metrics and checkpoint.
- [ ] Compare the exported web dataset against `SyntheticPeakDataset` at matching settings and seed.
- [ ] Deploy a *new* Streamlit test app and test its browser buttons, charts and downloads.
- [ ] Only after approval plan any changes to the existing online app/domain.
- [ ] For exact reproduction of the historic R² = 0.9980, rerun the full 50-epoch, 50,000-sample experiment and report hardware, library versions, and measured metrics. That full rerun is not included here.
- [ ] Add manuscript citation metadata (`CITATION.cff`, authors, publication DOI) after author list and DOI are finalized.
