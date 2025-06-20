import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# === Load and preprocess EEG data ===
eeg_df = pd.read_csv("rawBrainwaves_1746313503.csv")
eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
sfreq = 256

eeg_df = eeg_df[~(eeg_df[eeg_channels] == 0).any(axis=1)]
eeg_data = eeg_df[eeg_channels].to_numpy().T / 1e2  # adjust scaling if needed

# Create Raw object
info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(eeg_data, info)
raw_filtered = raw.copy().filter(1., 50.)

# === Compute PSD for topomap ===
psd = raw_filtered.compute_psd(fmin=4, fmax=7)  # for Theta
psd_values = psd.get_data().mean(axis=1)        # average across freqs

# === Add (approximate) standard montage === 
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False)

# === Plot topomap ===
mne.viz.plot_topomap(psd_values, raw.info, show=True, names=eeg_channels,
                     sphere=0.1, contours=0, cmap="plasma", res=128)
