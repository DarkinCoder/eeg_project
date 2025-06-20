import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# === Configuration ===
folder_path = "Brain Wave Files"
eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
sfreq = 256  # Muse headset sampling rate

# EEG bands definition
bands = {
    'Theta': (4, 7),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 50)
}

# Placeholder for all band powers
all_band_powers = []

# === Process each EEG CSV file ===
for i, file_name in enumerate(sorted(os.listdir(folder_path))):
    if not file_name.endswith(".csv"):
        continue
    
    file_path = os.path.join(folder_path, file_name)
    eeg_df = pd.read_csv(file_path)
    eeg_df = eeg_df[~(eeg_df[eeg_channels] == 0).any(axis=1)]  # remove zero rows
    eeg_data = eeg_df[eeg_channels].to_numpy().T / 1e2         # adjust scale

    # Create MNE raw object
    info = mne.create_info(ch_names=eeg_channels, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    raw_filtered = raw.copy().filter(2., 50.)

    # Compute PSD
    psd = raw_filtered.compute_psd(fmin=1, fmax=50, n_fft=512)
    psds = psd.get_data()
    freqs = psd.freqs

    # Compute relative power per band
    band_powers = {}
    total_power = 0
    for band, (fmin, fmax) in bands.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psds[:, band_mask].mean()
        band_powers[band] = band_power
        total_power += band_power

    # Normalize to relative power
    for band in bands:
        band_powers[band] = band_powers[band] / total_power if total_power > 0 else 0
    
    band_powers['Trial'] = i + 1
    all_band_powers.append(band_powers)

# === Create DataFrame of band powers ===
bandpower_df = pd.DataFrame(all_band_powers)
bandpower_df['Trial'] = range(1, len(bandpower_df) + 1)

# === Load subjective score file ===
performance_df = pd.read_excel("Performance Rating P01_7trials.xlsx")  # must have 'Trial' column

# === Merge on Trial ===
merged_df = pd.merge(bandpower_df, performance_df, on='Trial')

# === Compute correlation matrix ===
bands_only = ['Theta', 'Alpha', 'Beta', 'Gamma']
score_columns = [col for col in performance_df.columns if col != 'Trial']

correlation_matrix = pd.DataFrame(index=bands_only, columns=score_columns)

for band in bands_only:
    for score in score_columns:
        r, _ = spearmanr(merged_df[band], merged_df[score])
        correlation_matrix.loc[band, score] = r

correlation_matrix = correlation_matrix.astype(float)

# === Plot heatmap ===
plt.figure(figsize=(8, 5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Correlation between Relative EEG Band Power and Performance Metrics")
plt.tight_layout()
plt.show()
