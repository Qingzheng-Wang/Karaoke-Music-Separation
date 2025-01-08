# Use pitch estimation and NMF to separate vocals and instrumental. 
# from pitch import create_binary_mask, create_soft_mask 
from pitch_japan import *
import librosa
import numpy as np
import soundfile
from tqdm import tqdm
from sklearn.decomposition import NMF
from auv_hmm import *
import time
import matplotlib.pyplot as plt

def NMF_train(M, B_init, W_init, n_iter, typ='vocal'):
    """
    NMF train process.
    M = BW
    Args:
        - M: a non-negative matrix, (p, q).
        - B_init: a couple of initial non-negative base vectors, (p, k).
        - W_init: a initial weight matrix, (k, q).
        - n_iter: the number of iterations to train.
        - typ: the type of the matrix, 'vocal' or 'instr'.
    Returns:
        - B: the optimized base matrix, (p, k).
        - W: the optimized weight matrix, (k, q).
    """

    B = B_init
    W = W_init
    I_pq = np.ones_like(M)

    progress_bar = tqdm(range(n_iter), desc=f'NMF Training ({typ})', postfix={'KL Divergence': 0})
    for i in progress_bar:
        M_B_W = M / (B @ W + 2.2204e-16)
        B *= ((M_B_W @ W.T) / (I_pq @ W.T))
        W *= ((B.T @ M_B_W) / (B.T @ I_pq))
        div = KL_div(M, B @ W)

        progress_bar.set_postfix({'KL Divergence': div})
    
    return B, W

def NMF_train_mask(M, mask, B_init, W_init, n_iter, typ='vocal'):
    """
    NMF train process.
    M = BW
    Args:
        - M: a non-negative matrix, (p, q).
        - mask: a binary mask, (p, q).
        - B_init: a couple of initial non-negative base vectors, (p, k).
        - W_init: a initial weight matrix, (k, q).
        - n_iter: the number of iterations to train.
        - typ: the type of the matrix, 'vocal' or 'instr'.
    Returns:
        - B: the optimized base matrix, (p, k).
        - W: the optimized weight matrix, (k, q).
    """

    B = np.maximum(B_init, 2.2204e-16)
    W = np.maximum(W_init, 2.2204e-16)

    progress_bar = tqdm(range(n_iter), desc=f'NMF Training ({typ})', postfix={'KL Divergence': 0})
    for i in progress_bar:
        M_B_W = M / (np.maximum(B @ W, 2.2204e-16))
        B = B * (((mask * M_B_W) @ W.T) / (np.maximum(mask @ W.T, 2.2204e-16)))
        W = W * ((B.T @ (mask * M_B_W)) / (np.maximum(B.T @ mask, 2.2204e-16)))
        div = KL_div(M, B @ W)

        progress_bar.set_postfix({'KL Divergence': div})
    
    return B, W

def NMF_train_sklearn(M, n_components, n_iter, typ='vocal'):
    model = NMF(n_components=n_components, init='random', max_iter=n_iter, solver='mu', beta_loss='kullback-leibler', random_state=0)
    W = model.fit_transform(M)
    H = model.components_
    return W, H

def KL_div(A, B):
    A = np.maximum(A, 2.2204e-16)
    B = np.maximum(B, 2.2204e-16)
    return np.mean(A * np.log(A / B) - A + B)

def VAR(y, y_hat):
    # y: reference signal
    # y_hat: estimated signal
    return 10 * np.log10(np.sum(y**2) / np.sum((y - y_hat)**2))

def plot_mask(mask): 
    plt.imshow(mask, aspect='auto', origin='lower')
    plt.show()

def separate_hmm_nmf(wav_path, out_vocal_path, out_instr_path, bandwidth=50, num_partial_freq=60, num_components=8):
    y_0, sr = soundfile.read(wav_path)
    if y_0.ndim == 2:
        y_1 = y_0[:, 0]
        scale_instr = np.max(np.abs(y_1))
        y_1 = y_1 / scale_instr
        y_2 = y_0[:, 1]
        scale_vocal = np.max(np.abs(y_2))
        y_2 = y_2 / scale_vocal
        # y = y_1 * 1.5 + y_2 * 0.5
        snr = 4 # 20 log(y_1 / y_2) = snr, y_1 instr, y_2 vocal
        y = (y_1 * 10**(snr/20) + y_2) / (1 + 10**(snr/20))
        y = y / np.max(np.abs(y))
    else:
        y = y_0 / np.max(np.abs(y_0))
        scale_instr = np.max(np.abs(y_0))
        scale_vocal = np.max(np.abs(y_0))

    frame_length = int(sr * 0.040)
    hop_length= int(sr * 0.020)

    # ============= GMMHMM predict the vocal frames =============
    train_mix_mfcc, auv_labels = dataset_train_hmm()
    # normalize the features
    train_mix_mfcc = (train_mix_mfcc - np.mean(train_mix_mfcc, axis=0, keepdims=True)) / np.std(train_mix_mfcc, axis=0, keepdims=True)
    model = train_hmm(train_mix_mfcc, n_iter=100)
    pred_states_train = model.predict(train_mix_mfcc.T)
    state_label_map = map_states_to_labels(pred_states_train, auv_labels, model.n_components)

    y_mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=39, n_fft=frame_length, hop_length=hop_length, win_length=frame_length)
    y_mfcc = (y_mfcc - np.mean(y_mfcc, axis=0, keepdims=True)) / np.std(y_mfcc, axis=0, keepdims=True)
    pred_states = model.predict(y_mfcc.T)
    pred_labels = np.array([state_label_map[state] for state in pred_states]) # 0 for unvocal, 1 for vocal

    # ============= Create Mask =============
    mask = create_binary_mask(y, sr, pred_labels, frame_length, hop_length, bandwidth, num_partial_freq) # 0 for vocal, 1 for instrumental
    plot_mask(mask)

    # ============= NMF =============
    spec = librosa.stft(y, n_fft=frame_length, hop_length=hop_length) # (F, T)
    mag = np.abs(spec)
    phase = np.angle(spec)
    mag = (mag - np.mean(mag)) / (np.std(mag) + 2.2204e-16)
    mag_scale = np.mean(mag) / num_components

    mag_instr = mag * mask # (F, T)
    # B_init = np.random.rand(mag_instr.shape[0], num_components) * mag_scale + 2.2204e-16
    B_init = np.load("E:\\24_fall_backup\\midterm_model_n_code_1\\model\\B_instr.npy")
    W_init = np.random.rand(num_components, mag_instr.shape[1]) * mag_scale + 2.2204e-16
    
    # B_instr, W_instr = NMF_train_sklearn(mag_instr, n_components=20, n_iter=100)
    B_instr, W_instr = NMF_train_mask(mag, mask, B_init, W_init, n_iter=100, typ='mask')

    mag_vocal = np.maximum(mag - B_instr @ W_instr, 0) * (np.ones_like(mask) - mask)
    mag_instr_rec = mag - mag_vocal

    stft_vocal = mag_vocal * phase
    stft_instr_rec = mag_instr_rec * phase
    vocal = librosa.istft(stft_vocal, hop_length=hop_length, n_fft=frame_length, length=len(y))
    instr = librosa.istft(stft_instr_rec, hop_length=hop_length, n_fft=frame_length, length=len(y))

    # rescale to the original scale
    vocal = vocal / np.max(np.abs(vocal)) * scale_vocal
    instr = instr / np.max(np.abs(instr)) * scale_instr

    if y_0.ndim == 2:
        VAR_vocal = VAR(y_2, vocal)
        VAR_instr = VAR(y_1, instr)
        print(f"VAR for Vocal: {VAR_vocal} dB")
        print(f"VAR for Instr: {VAR_instr} dB")

    soundfile.write(out_vocal_path, vocal, sr)
    soundfile.write(out_instr_path, instr, sr)

def separate(wav_path, out_vocal_path, out_instr_path, bandwidth=50, num_partial_freq=60, num_components=8):
    y_0, sr = soundfile.read(wav_path)
    if y_0.ndim == 2:
        y_1 = y_0[:, 0]
        scale_instr = np.max(np.abs(y_1))
        y_1 = y_1 / scale_instr
        y_2 = y_0[:, 1]
        scale_vocal = np.max(np.abs(y_2))
        y_2 = y_2 / scale_vocal
        snr = 4 # 20 log(y_1 / y_2) = snr, y_1 instr, y_2 vocal
        y = (y_1 * 10**(snr/20) + y_2) / (1 + 10**(snr/20))
        soundfile.write("E:\\24_fall_backup\\midterm_model_n_code_1\\results_pitch\\abjones_1_03.wav", y, sr)
        y = y / np.max(np.abs(y))
    else:
        y = y_0 / np.max(np.abs(y_0))
        scale_instr = np.max(np.abs(y_0))
        scale_vocal = np.max(np.abs(y_0))

    frame_length = int(sr * 0.04)
    hop_length= int(sr * 0.02)

    mask = create_binary_mask(y, sr, frame_length, hop_length, bandwidth, num_partial_freq) # 0 for vocal, 1 for instrumental
    plot_mask(mask)

    spec = librosa.stft(y, n_fft=frame_length, hop_length=hop_length) # (F, T)
    mag = np.abs(spec)
    phase = np.angle(spec)
    mag_scale = np.mean(mag) / num_components

    mag_instr = mag * mask # (F, T)
    B_init = np.random.rand(mag_instr.shape[0], num_components) * mag_scale + 2.2204e-16
    W_init = np.random.rand(num_components, mag_instr.shape[1]) * mag_scale + 2.2204e-16
    
    B_instr, W_instr = NMF_train_mask(mag, mask, B_init, W_init, n_iter=30, typ='mask')

    mag_vocal = np.maximum(mag - B_instr @ W_instr, 0) * (np.ones_like(mask) - mask)
    mag_instr_rec = mag - mag_vocal

    stft_vocal = mag_vocal * phase
    stft_instr_rec = mag_instr_rec * phase
    vocal = librosa.istft(stft_vocal, hop_length=hop_length, n_fft=frame_length, length=len(y))
    instr = librosa.istft(stft_instr_rec, hop_length=hop_length, n_fft=frame_length, length=len(y))

    # rescale to the original scale
    vocal = vocal / np.max(np.abs(vocal)) * scale_vocal
    instr = instr / np.max(np.abs(instr)) * scale_instr

    if y_0.ndim == 2:
        VAR_vocal = VAR(y_2, vocal)
        VAR_instr = VAR(y_1, instr)
        print(f"VAR for Vocal: {VAR_vocal} dB")
        print(f"VAR for Instr: {VAR_instr} dB")

    soundfile.write(out_vocal_path, vocal, sr)
    soundfile.write(out_instr_path, instr, sr)

def separate_given_pitch(wav_path, out_vocal_path, out_instr_path, pitch_path, bandwidth=50, num_partial_freq=60, num_components=8):
    y_0, sr = soundfile.read(wav_path)
    if y_0.ndim == 2:
        y_1 = y_0[:, 0]
        scale_instr = np.max(np.abs(y_1))
        y_1 = y_1 / scale_instr
        y_2 = y_0[:, 1]
        scale_vocal = np.max(np.abs(y_2))
        y_2 = y_2 / scale_vocal
        snr = 4 # 20 log(y_1 / y_2) = snr, y_1 instr, y_2 vocal
        y = (y_1 * 10**(snr/20) + y_2) / (1 + 10**(snr/20))
        y = y / np.max(np.abs(y))
    else:
        y = y_0 / np.max(np.abs(y_0))
        scale_instr = np.max(np.abs(y_0))
        scale_vocal = np.max(np.abs(y_0))

    frame_length = int(sr * 0.04)
    hop_length= int(sr * 0.02)

    pitch = np.loadtxt(pitch_path)
    mask = create_binary_mask_given_pitch(y, sr, pitch, frame_length, hop_length, bandwidth, num_partial_freq) # 0 for vocal, 1 for instrumental
    # mask = create_soft_mask(y, sr, frame_length, hop_length, bandwidth)
    plot_mask(mask)

    spec = librosa.stft(y, n_fft=frame_length, hop_length=hop_length) # (F, T)
    mag = np.abs(spec)
    # mag = (mag - np.mean(mag)) / (np.std(mag) + 2.2204e-16)
    phase = np.angle(spec)
    mag_scale = np.mean(mag) / num_components

    mag_instr = mag * mask # (F, T)
    B_init = np.random.rand(mag_instr.shape[0], num_components) * mag_scale + 2.2204e-16
    # B_init = np.load("E:\\24_fall_backup\\midterm_model_n_code_1\\model\\B_instr.npy")
    W_init = np.random.rand(num_components, mag_instr.shape[1]) * mag_scale + 2.2204e-16
    
    # B_instr, W_instr = NMF_train_sklearn(mag_instr, n_components=20, n_iter=1000)
    B_instr, W_instr = NMF_train_mask(mag, mask, B_init, W_init, n_iter=10, typ='mask')

    mag_vocal = np.maximum(mag - B_instr @ W_instr, 0) * (np.ones_like(mask) - mask)
    mag_instr_rec = mag - mag_vocal

    stft_vocal = mag_vocal * phase
    stft_instr_rec = mag_instr_rec * phase
    vocal = librosa.istft(stft_vocal, hop_length=hop_length, n_fft=frame_length, length=len(y))
    instr = librosa.istft(stft_instr_rec, hop_length=hop_length, n_fft=frame_length, length=len(y))

    # rescale to the original scale
    vocal = vocal / np.max(np.abs(vocal)) * scale_vocal
    instr = instr / np.max(np.abs(instr)) * scale_instr

    if y_0.ndim == 2:
        VAR_vocal = VAR(y_2, vocal)
        VAR_instr = VAR(y_1, instr)
        print(f"VAR for Vocal: {VAR_vocal} dB")
        print(f"VAR for Instr: {VAR_instr} dB")

    soundfile.write(out_vocal_path, vocal, sr)
    soundfile.write(out_instr_path, instr, sr)

if __name__ == "__main__":
    file_name = "abjones_1_03"

    if file_name == "wubai":
        wubai_instr_path = "E:\\24_fall_backup\\midterm_model_n_code_1\\dataset\\wubai_instr.wav"
        wubai_vocal_path = "E:\\24_fall_backup\\midterm_model_n_code_1\\dataset\\wubai_vocal.wav"
        wubai_instr, sr = soundfile.read(wubai_instr_path)
        wubai_vocal, sr = soundfile.read(wubai_vocal_path)
        wubai = []
        wubai.append(wubai_instr)
        wubai.append(wubai_vocal)
        wubai = np.array(wubai).T # (T, 2)
        wubai_remix_path = "E:\\24_fall_backup\\midterm_model_n_code_1\\dataset\\wubai_remix.wav"
        soundfile.write(wubai_remix_path, wubai, sr)
        file_name = "wubai_remix"
    
    wav_path = f"E:\\24_fall_backup\\midterm_model_n_code_1\\dataset\\{file_name}.wav"
    pitch_path = f"E:\\24_fall_backup\\midterm_model_n_code_1\\dataset\\MIR-1k\\PitchLabel\\{file_name}.pv"

    current_time = time.strftime("%Y%m%d-%H%M%S")
    out_vocal_path = f"E:\\24_fall_backup\\midterm_model_n_code_1\\results_pitch\\{file_name}_vocal_{current_time}.wav"
    out_instr_path = f"E:\\24_fall_backup\\midterm_model_n_code_1\\results_pitch\\{file_name}_instr_{current_time}.wav"
    # separate_hmm_nmf(wav_path, out_vocal_path, out_instr_path)
    separate(wav_path, out_vocal_path, out_instr_path)
    # separate_given_pitch(wav_path, out_vocal_path, out_instr_path, pitch_path)