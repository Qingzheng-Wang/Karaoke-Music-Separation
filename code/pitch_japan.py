import numpy as np
from scipy.signal import find_peaks, gaussian, convolve
from scipy.fftpack import fft
import librosa

# Step 1: Note Onset Detection
def detect_onsets(audio, sr, low_pass_freq=500, frame_size=0.006, overlap=0.5):
    """
    Detect note onsets in a bass line.
    """
    frame_samples = int(frame_size * sr)
    step = int(frame_samples * (1 - overlap))

    # Low-pass filter: retain frequencies below `low_pass_freq`
    low_pass_filter = np.fft.rfftfreq(frame_samples, d=1/sr) <= low_pass_freq

    power = []
    for start in range(0, len(audio) - frame_samples, step):
        frame = audio[start:start+frame_samples]
        spectrum = np.abs(fft(frame))[:len(low_pass_filter)]
        filtered_power = np.sum(spectrum[low_pass_filter])
        power.append(filtered_power)

    power = np.array(power)
    smoothed_power = convolve(power, gaussian(11, std=2), mode='same')
    peaks, _ = find_peaks(smoothed_power, height=np.max(smoothed_power)/66)

    return peaks * step / sr  # Convert indices to seconds

# Step 2: Frequency Estimation and Hypothesis Generation
def estimate_frequencies(audio, sr, onsets, f0_max=200):
    """
    Estimate fundamental frequencies (F0) based on detected onsets.
    """
    hypotheses = []
    for onset in onsets:
        start = int(onset * sr)
        end = min(len(audio), start + int(0.1 * sr))  # Use 100 ms window
        frame = audio[start:end]
        spectrum = np.abs(fft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1/sr)

        # Hypothesize frequencies below `f0_max`
        candidates = freqs[freqs <= f0_max]
        powers = spectrum[:len(candidates)]
        if len(powers) > 0:
            f0 = candidates[np.argmax(powers)]
            hypotheses.append((onset, f0))

    return hypotheses

def estimate_frequencies_with_salience(audio, sr, onsets, f0_max=200):
    hypotheses = []
    for onset in onsets:
        start = int(onset * sr)
        end = min(len(audio), start + int(0.1 * sr))  # Use 100 ms window
        frame = audio[start:end]
        spectrum = np.abs(fft(frame))
        freqs = np.fft.rfftfreq(len(frame), d=1/sr)

        # Hypothesize frequencies below `f0_max`
        candidates = freqs[freqs <= f0_max]
        powers = spectrum[:len(candidates)]

        # Salience function and local maxima
        salience = powers  # Example: use magnitude as salience
        peaks, _ = find_peaks(salience, height=np.max(salience) * 0.8)

        if len(peaks) > 0:
            f0 = candidates[peaks[np.argmax(salience[peaks])]]
            hypotheses.append((onset, f0))

    return hypotheses

# Step 3: Hypothesis Tracking
def track_hypotheses(hypotheses, tolerance=0.03):
    """
    Track hypotheses over time.
    """
    tracked = []
    for onset, f0 in hypotheses:
        # Assume a simple tracker here; improvements can be added
        tracked.append((onset, f0))

    return tracked

# Step 4: Hypothesis Trimming
def resolve_hypotheses(tracked):
    """
    Resolve overlapping hypotheses and finalize transcription.
    """
    resolved = []
    for onset, f0 in tracked:
        resolved.append((onset, f0))

    return resolved


def pitch_estimation(audio, sr, resolved, frame_length=2048, hop_length=512):
    num_frames = int(1 + (len(audio) - frame_length) // hop_length)

    pitch_estimates = []

    for onset, f0 in resolved:
        frame_idx = int(onset * sr // hop_length)
        if frame_idx < num_frames and f0 > 0:
            pitch_estimates.append((frame_idx, f0))

    return pitch_estimates

# Binary mask creation
def create_binary_mask(y, sr, frame_length=1746, hop_length=882, bandwidth=50, num_partial_freq=60, pred_labels=None):

    onsets = detect_onsets(y, sr)
    hypotheses = estimate_frequencies(y, sr, onsets)
    tracked = track_hypotheses(hypotheses)
    resolved = resolve_hypotheses(tracked)
    pitch_estimates = pitch_estimation(y, sr, resolved, frame_length, hop_length) # (frame index, pitch (Hz))

    # Compute STFT
    spec = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, window='hann')
    mag = np.abs(spec)
    mag = mag / np.max(mag)

    freqs = np.linspace(0, sr / 2, frame_length // 2 + 1)

    # Create binary mask
    F, T = mag.shape
    binary_mask = np.ones((F, T), dtype=int)

    for m, pitch in pitch_estimates:
        if pitch > 0 and (pred_labels is None or pred_labels[m] == 1):  # Voiced frame
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

def create_binary_mask_given_pitch(y, sr, pitch_given, frame_length=1746, hop_length=882, bandwidth=50, num_partial_freq=60, pred_labels=None):

    # Compute STFT
    spec = librosa.stft(y, n_fft=frame_length, hop_length=hop_length, window='hann')
    mag = np.abs(spec)
    mag = mag / np.max(mag)

    freqs = np.linspace(0, sr / 2, frame_length // 2 + 1)

    # Create binary mask
    F, T = mag.shape
    binary_mask = np.ones((F, T), dtype=int)

    for m, pitch in enumerate(pitch_given):
        if pitch > 0 and (pred_labels is None or pred_labels[m] == 1):  # Voiced frame
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