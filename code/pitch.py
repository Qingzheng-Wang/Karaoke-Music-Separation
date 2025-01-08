import numpy as np
import librosa

# Placeholder for Ryynänen and Klapuri's melody transcription algorithm
def pitch_estimation(y, sr, frame_length, hop_length=882):
    # Implement or use a library for Ryynänen and Klapuri [7]
    # For now, we use librosa's pitch tracking as a placeholder.
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)

    pitch_estimates = []

    for frame in range(pitches.shape[1]):
        # Find local maxima in the salience function (magnitude)
        indices = np.where(magnitudes[:, frame] > np.max(magnitudes[:, frame]) * 0.8)[0]
        if len(indices) > 0:
            pitch_estimates.append(np.mean(pitches[indices, frame]))
        else:
            pitch_estimates.append(0)  # Unvoiced frame

    return np.array(pitch_estimates)

# Binary mask creation
def create_binary_mask(y, sr, frame_length=1746, hop_length=882, bandwidth=50, num_partial_freq=30):
    # Compute STFT
    spec = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, window='hann')
    mag = np.abs(spec)
    mag = mag / np.max(mag)

    freqs = np.linspace(0, sr / 2, frame_length // 2 + 1)

    # Estimate pitches
    pitch_estimates = pitch_estimation(y, sr, frame_length, hop_length)

    # Create binary mask
    F, T = mag.shape
    binary_mask = np.ones((F, T), dtype=int)

    for m, pitch in enumerate(pitch_estimates):
        if pitch > 0:  # Voiced frame
            # partial_freqs = np.arange(pitch, sr / 2, pitch)
            partial_freqs = pitch * np.arange(1, num_partial_freq + 1)

            for f in partial_freqs:
                if f > sr / 2:
                    break
                lower_bound = f - bandwidth / 2
                upper_bound = f + bandwidth / 2

                # Find frequency bins within the bandwidth
                bins = np.where((freqs >= lower_bound) & (freqs <= upper_bound))[0]

                binary_mask[bins, m] = 0 # 0 for vocal, 1 for unvocal

    return binary_mask


# Soft mask creation
def create_soft_mask(y, sr, frame_length=1746, hop_length=882, bandwidth=50):
    # Compute STFT
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, window='hann')
    magnitude = np.abs(D)
    freqs = np.linspace(0, sr / 2, frame_length // 2 + 1)

    # Estimate pitches
    pitch_estimates = pitch_estimation(y, sr, frame_length, hop_length)

    # Create soft mask
    K, M = magnitude.shape
    soft_mask = np.ones((K, M), dtype=float)

    for m, pitch in enumerate(pitch_estimates):
        if pitch > 0:  # Voiced frame
            partial_frequencies = np.arange(pitch, sr / 2, pitch)

            for f in partial_frequencies:
                lower_bound = f - bandwidth / 2
                upper_bound = f + bandwidth / 2

                # Find frequency bins within the bandwidth
                bins = np.where((freqs >= lower_bound) & (freqs <= upper_bound))[0]

                # Apply a soft Gaussian weighting to the bins
                for b in bins:
                    soft_mask[b, m] *= np.exp(-((freqs[b] - f) ** 2) / (2 * (bandwidth / 2) ** 2))

    # Normalize the soft mask to be within [0, 1]
    soft_mask = soft_mask / np.max(soft_mask)

    return soft_mask