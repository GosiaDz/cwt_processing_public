from pathlib import Path
import sys
import numpy as np
import torch
from cwt_processing.experiment_engine import EnginePeakNetwork,MultiWaveletPreprocessing
from cwt_processing.synthetic_peak_generator import SyntheticPeakDataset
from transformer.predict_checkpoint import predict

def test_inference_tensor_shape():
    pre=MultiWaveletPreprocessing({'MH':[5.0,23.0],'GD':[5.0,23.0]})
    model=EnginePeakNetwork(preprocessing=pre,signal_length=400,n_outputs=6,
        target_mean=torch.zeros(6),target_std=torch.ones(6))
    signals=np.stack([SyntheticPeakDataset(3)[i][0].numpy() for i in range(3)])
    res=predict(model,signals,batch_size=2)
    assert res.shape==(3,6)
    assert np.all(np.isfinite(res))
