"""Important regression tests: the web interface cannot silently diverge from training."""
import io
import zipfile
import numpy as np
import pytest
from cwt_processing.synthetic_peak_generator import SyntheticPeakDataset
from web_generator.bridge import generate_chunk,combine_chunks,zip_dataset

@pytest.mark.parametrize('family',['gaussian','lorentzian','pseudovoigt','paired_mix','independent_mix'])
@pytest.mark.parametrize('noise',[0.0,0.02])
def test_bit_for_bit_same_as_transformer(family,noise):
    native=SyntheticPeakDataset(9,family=family,seed=31337,noise_level=noise)
    web=combine_chunks([generate_chunk(native,0,4),generate_chunk(native,4,9)])
    for i in range(9):
        x,y=native[i]
        assert np.array_equal(web.summed[i],x.numpy())
        assert np.allclose(web.peak1[i]+web.peak2[i],x.numpy(),atol=1e-6) if not noise else True
        expected=y.numpy()
        row=web.labels.iloc[i]
        reconstructed=np.asarray([[row.mu1,row.A1,row.sigma1,row.eta1],
                                  [row.mu2,row.A2,row.sigma2,row.eta2]],dtype=np.float32)
        assert np.array_equal(reconstructed,expected)
    with zipfile.ZipFile(io.BytesIO(zip_dataset(web)),'r') as z:
        with np.load(io.BytesIO(z.read('centered_peaks/spectra.npz'))) as data:
            assert np.array_equal(data['summed'],web.summed)
            assert data['targets'].shape==(9,2,4)

def test_center_is_fixed():
    ds=SyntheticPeakDataset(12,seed=31337)
    for i in range(12):
        target=ds[i][1].numpy()
        assert abs(float(target[0,0]+target[1,0])-400)<0.0001
