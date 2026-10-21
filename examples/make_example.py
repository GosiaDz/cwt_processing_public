"""Regenerate the example using EXACTLY the dataset the Transformer trains on."""
from __future__ import annotations
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for folder in (ROOT,ROOT/'transformer'/'src'):
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
import json
import numpy as np
from cwt_processing.synthetic_peak_generator import SyntheticPeakDataset
from web_generator.bridge import generate_chunk

native=SyntheticPeakDataset(12,family='gaussian',n_points=400,x_min=0.,x_max=400.,
                             center_mode='fixed',seed=31337,noise_level=0.)
result=generate_chunk(native,0,len(native))
file=Path(__file__).resolve().parent/'publication_gaussian_seed31337_N12.npz'
targets=result.labels[['mu1','A1','sigma1','eta1','mu2','A2','sigma2','eta2']].to_numpy('float32').reshape(-1,2,4)
np.savez_compressed(file,x_axis=result.x,summed=result.summed,
                    peak1=result.peak1,peak2=result.peak2,targets=targets)
result.labels.to_csv(file.with_suffix('.csv'),index=False)
file.with_suffix('.json').write_text(json.dumps(result.config,indent=2),encoding='utf-8')
print('Wrote',file)
