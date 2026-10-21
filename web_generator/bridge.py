"""Streamlit adapter: sampling is performed only by SyntheticPeakDataset."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import json
import re
import zipfile
import numpy as np
import pandas as pd
from cwt_processing.synthetic_peak_generator import SyntheticPeakDataset

@dataclass
class GeneratedData:
    x: np.ndarray
    peak1: np.ndarray
    peak2: np.ndarray
    summed: np.ndarray
    labels: pd.DataFrame
    config: dict

def generate_chunk(ds: SyntheticPeakDataset, start: int, end: int) -> GeneratedData:
    if not 0 <= start < end <= len(ds):
        raise ValueError('Invalid sample interval')
    p1, p2, y, labels = [], [], [], []
    for i in range(start,end):
        # The exact training signal, including its own RNG-driven noise.
        spectrum, truth = ds[i]
        # Reconstruct analytically *without modifying* native signal/targets.
        g = ds._sample_geometry(ds._rng(i))
        p1.append(ds._component(g.family1,g.mu1,g.amplitude1,g.sigma1,g.eta1).astype('float32'))
        p2.append(ds._component(g.family2,g.mu2,g.amplitude2,g.sigma2,g.eta2).astype('float32'))
        y.append(spectrum.numpy())
        t=truth.numpy()
        labels.append(dict(sample_index=i,mu1=float(t[0,0]),A1=float(t[0,1]),
                           sigma1=float(t[0,2]),eta1=float(t[0,3]),family1=g.family1,
                           mu2=float(t[1,0]),A2=float(t[1,1]),
                           sigma2=float(t[1,2]),eta2=float(t[1,3]),family2=g.family2))
    attrs = ['n_samples','family','n_points','x_min','x_max','center_mode','delta_mu_min',
             'delta_mu_max','sigma_min','sigma_max','sigma_min_frac','sigma_max_frac',
             'sigma_ratio_min','sigma_ratio_max','delta_width_min','delta_width_max',
             'amplitude_high_min','amplitude_high_max','amplitude_ratio_min',
             'amplitude_ratio_max','eta_min','eta_max','edge_sigma_margin',
             'noise_level','seed','distinct_epochs','max_attempts']
    return GeneratedData(ds._x.copy(),np.stack(p1),np.stack(p2),np.stack(y),
                         pd.DataFrame(labels),{a:getattr(ds,a) for a in attrs})

def combine_chunks(chunks: list[GeneratedData]) -> GeneratedData:
    if not chunks or any(not np.array_equal(chunks[0].x,c.x) for c in chunks):
        raise ValueError('Chunks need the same axis')
    first = chunks[0]
    return GeneratedData(first.x,*[np.concatenate([getattr(c,a) for c in chunks])
                                   for a in ['peak1','peak2','summed']],
                         pd.concat([c.labels for c in chunks],ignore_index=True),first.config)

def zip_dataset(data:GeneratedData,name='centered_peaks') -> bytes:
    safe = re.sub('[^A-Za-z0-9_.-]+','_',name).strip('._') or 'centered_peaks'
    arr=BytesIO()
    targets=data.labels[['mu1','A1','sigma1','eta1','mu2','A2','sigma2','eta2']].to_numpy(dtype='float32').reshape(-1,2,4)
    np.savez_compressed(arr,x_axis=data.x,peak1=data.peak1,peak2=data.peak2,
                        summed=data.summed,targets=targets)
    with BytesIO() as b:
        with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr(f'{safe}/spectra.npz',arr.getvalue())
            z.writestr(f'{safe}/labels.csv',data.labels.to_csv(index=False).encode())
            z.writestr(f'{safe}/config.json',json.dumps(data.config,indent=2))
            z.writestr(f'{safe}/README.txt','summed is native SyntheticPeakDataset output.\n'
                       'peak1 and peak2 are noiseless theoretical components.\n'
                       'targets (N,2,4): mu,A,sigma,eta.\n'
                       'Noise added to sum once: noise_level * max(sum) is its standard deviation.\n')
        return b.getvalue()
