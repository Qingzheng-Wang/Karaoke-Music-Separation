import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def extract_mfcc_mix(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, mono=False)
    y = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
    sr = 16000
    instr = y[0]
    vocal = y[1]
    mfcc_instr = librosa.feature.mfcc(
        y=instr, sr=sr, n_mfcc=n_mfcc, n_fft=int(sr * 0.032), 
        hop_length=int(sr * 0.008), win_length=int(sr * 0.032) 
    )
    mfcc_vocal = librosa.feature.mfcc(
        y=vocal, sr=sr, n_mfcc=n_mfcc, n_fft=int(sr * 0.032), 
        hop_length=int(sr * 0.008), win_length=int(sr * 0.032)
    )
    return mfcc_instr, mfcc_vocal

def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, mono=True)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=int(sr * 0.032), 
        hop_length=int(sr * 0.008), win_length=int(sr * 0.032) 
    )
    return mfcc

def plot_mfcc(mfcc_instr, mfcc_vocal, mfcc_sep_instr, mfcc_sep_vocal):
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.4)  # Add space between rows
    librosa.display.specshow(mfcc_instr, x_axis='time', ax=ax[0, 0])
    ax[0, 0].set(title='Target Instrument MFCC')
    librosa.display.specshow(mfcc_vocal, x_axis='time', ax=ax[0, 1])
    ax[0, 1].set(title='Target Vocal MFCC')
    librosa.display.specshow(mfcc_sep_instr, x_axis='time', ax=ax[1, 0])
    ax[1, 0].set(title='Separated Instrument MFCC')
    librosa.display.specshow(mfcc_sep_vocal, x_axis='time', ax=ax[1, 1])
    ax[1, 1].set(title='Separated Vocal MFCC')
    plt.show()

# Example usage
mix_path = "../dataset/khair_1_01.wav"
sep_instr_path = "../results/instr_rec_khair_1_01.wav"
sep_vocal_path = "../results/vocal_rec_khair_1_01.wav"

mfcc_instr, mfcc_vocal = extract_mfcc_mix(mix_path)
mfcc_spep_instr = extract_mfcc(sep_instr_path)
mfcc_spep_vocal = extract_mfcc(sep_vocal_path)
plot_mfcc(mfcc_instr, mfcc_vocal, mfcc_spep_instr, mfcc_spep_vocal)