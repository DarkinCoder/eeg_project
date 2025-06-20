import pandas as pd
import numpy as np
import os
import mne
from scipy.stats import sem
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

folder_path = "Brain Wave Files"
eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
sfreq = 256  # Muse default
theta_band = (4,7)

theta_curves = {}
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    eeg_df = pd.read_csv(file_path)
    
    eeg_df = eeg_df[~(eeg_df[eeg_channels] == 0).any(axis=1)]
    eeg_data = eeg_df[eeg_channels].to_numpy().T  # scale to volts for MNE

    # === Create MNE Raw object ===
    info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    raw_filtered = raw.copy().filter(1., 50.)

    # === Segment into non-overlapping 0.5-second epochs ===
    epochs = mne.make_fixed_length_epochs(raw_filtered, duration=1.0, preload=True)
    
    # === Compute Power Spectral Density (PSD) per epoch ===
    psd = epochs.compute_psd(method="welch", fmin=1, fmax=50, n_fft=256)
    psds = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)
    freqs = psd.freqs

    # === Extract theta frequency bands ===
    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    theta_power = psds[:, :, theta_mask].mean(axis=2)
    theta_power = theta_power.mean(axis=1)

    # Save in dict
    label = os.path.splitext(file_name)[0]
    theta_curves[label] = theta_power

# Align all theta curves by truncating to the shortest
min_len = min(len(p) for p in theta_curves.values())
aligned = np.array([p[:min_len] for p in theta_curves.values()])

mean_theta = aligned.mean(axis=0)
std_theta = aligned.std(axis=0)
stderr_theta = sem(aligned, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(mean_theta, label='Mean Theta', color='black')
plt.fill_between(range(min_len), mean_theta - std_theta, mean_theta + std_theta,
                 alpha=0.3, label='±1 SD', color='gray')
plt.title("Average Theta Power Across All Sessions")
plt.xlabel("Epoch (0.5s)")
plt.ylabel("Theta Power (µV²/Hz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
