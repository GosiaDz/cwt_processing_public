"""Load a trained experiment checkpoint and predict peak parameters from an NPZ file.

Example: python transformer/predict_checkpoint.py --run outputs/gauss --input examples/publication_gaussian_seed31337_N12.npz
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
import torch
from cwt_processing.experiment_engine import EnginePeakNetwork, MultiWaveletPreprocessing
from cwt_processing.synthetic_peak_generator import output_names


def restore_model(run: Path, checkpoint: Path | None = None) -> tuple[EnginePeakNetwork, dict]:
    cfg = json.loads((run/'resolved_config.json').read_text(encoding='utf-8'))
    if checkpoint is None:
        candidates = sorted(run.glob('best-*.ckpt'))
        if not candidates:
            candidates = sorted(run.glob('last.ckpt'))
        if not candidates:
            raise FileNotFoundError(f'No best-*.ckpt or last.ckpt in {run}')
        checkpoint = candidates[0]
    pre = MultiWaveletPreprocessing(cfg['resolved_wavelet_scales'],
                                    complex_mode=cfg['complex_mode'])
    model = EnginePeakNetwork(preprocessing=pre,signal_length=cfg['n_points'],
        n_outputs=cfg['n_outputs'],target_mean=torch.tensor(cfg['target_mean']),
        target_std=torch.tensor(cfg['target_std']),patch_size=cfg['patch_size'],
        d_model=cfg['d_model'],nhead=cfg['heads'],num_layers=cfg['layers'],
        dim_feedforward=cfg['ff_dim'],dropout=cfg['dropout'],lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],scheduler=cfg['scheduler'],
        scheduler_patience=cfg['scheduler_patience'],scheduler_factor=cfg['scheduler_factor'],
        min_lr=cfg['min_lr'])
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict'],strict=True)
    model.eval()
    return model,cfg

@torch.no_grad()
def predict(model: EnginePeakNetwork, signals: np.ndarray, batch_size:int=256) -> np.ndarray:
    if signals.ndim!=2:raise ValueError('Expect spectra of shape (samples,points)')
    predictions=[]
    for start in range(0,len(signals),batch_size):
        t=torch.as_tensor(signals[start:start+batch_size],dtype=torch.float32)
        raw,ctx=model(t)
        predictions.append(model.decode_predictions(raw,ctx).cpu().numpy())
    return np.concatenate(predictions)

def main():
    p=argparse.ArgumentParser(description='Predict peak parameters from an engine checkpoint')
    p.add_argument('--run',type=Path,required=True,help='Folder with resolved_config.json and checkpoint')
    p.add_argument('--input',type=Path,required=True,help='NPZ with summed or spectra array')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=Path('predictions.csv'))
    p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args()
    model,cfg=restore_model(a.run,a.checkpoint)
    with np.load(a.input) as source:
        key=next((k for k in ('summed','spectra') if k in source),None)
        if key is None:raise ValueError('NPZ lacks summed or spectra array')
        x=source[key]
    if x.shape[1]!=cfg['n_points']:raise ValueError('Number of signal points does not match checkpoint')
    out=predict(model,x,a.batch_size)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(output_names(cfg['n_outputs']));writer.writerows(out)
    print(f'Saved {len(out)} predictions to {a.output}')
if __name__=='__main__':main()
