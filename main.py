from mne.datasets import sample
import mne
data_path = sample.data_path()
raw = mne.io.read_raw_fif(str(data_path) + '/MEG/sample/sample_audvis_raw.fif', preload=True)
raw.plot()
