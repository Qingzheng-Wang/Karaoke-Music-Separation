import numpy as np
from tqdm import tqdm
import librosa
import os
import soundfile
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

results_dir = config['results_dir']

def NMF_train(M, B_init, W_init, n_iter, typ='vocal'):
    """
    NMF train process.
    M = BW
    Args:
        - M: a non-negative matrix, (p, q).
        - B_init: a couple of initial non-negative base vectors, (p, k).
        - W_init: a initial weight matrix, (k, p).
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

def KL_div(A, B):
    A = np.maximum(A, 2.2204e-16)
    B = np.maximum(B, 2.2204e-16)
    return np.sum(A * np.log(A / B) - A + B)

def train(train_vocal_feat, train_instr_feat, num_bases=256, num_iters=200):
    B_init = np.random.rand(train_vocal_feat.shape[0], num_bases) + 2.2204e-16
    W_init = np.random.rand(num_bases, train_vocal_feat.shape[1]) + 2.2204e-16

    B_vocal, W_vocal = NMF_train(train_vocal_feat, B_init, W_init, num_iters, 'vocal')
    B_instr, W_instr = NMF_train(train_instr_feat, B_init, W_init, num_iters, 'instr')
    div_vocal = KL_div(train_vocal_feat, B_vocal @ W_vocal)
    div_instr = KL_div(train_instr_feat, B_instr @ W_instr)

    print(f"KL Divergence for Vocal ({num_iters}): {div_vocal}")
    print(f"KL Divergence for Instr ({num_iters}): {div_instr}")

    return B_vocal, B_instr


def separate(M_mixed, B_vocal, B_instr, num_iters=64):
    """
    Separate mixed signal given the bases of vocal and instr. 
    Args:
        - test_mix_feat: mixed signal's feat, (p, q).
        - B_vocal: bases of vocal, (p, k).
        - B_instr: bases of instrumental, (p, k).
        - num_iters: the number of iterations to train.
    Returns:
        - mag_vocal_rec: the recovery vocal spectrogram, (p, q).
        - mag_instr_rec: the recovery instr spectrogram, (p, q).
    """
    B = np.hstack((B_vocal, B_instr)) # (p, 2k)
    k = int(B.shape[1] / 2)
    W = np.random.rand(2 * k, M_mixed.shape[1]) + 2.2204e-16
    I_pq = np.ones_like(M_mixed)

    for i in range(num_iters):
        M_B_W = M_mixed / (B @ W)
        W = W * ((B.T @ M_B_W) / (B.T @ I_pq))

    W_vocal = W[:k, :]
    W_instr = W[k:, :]

    mag_vocal_rec = B_vocal @ W_vocal
    mag_instr_rec = B_instr @ W_instr

    return mag_vocal_rec, mag_instr_rec

def test(test_feats, test_phases, test_files, B_vocal, B_instr, num_iters=64):
    # remove the root dir in test_files
    test_files = [file.split('\\')[-1].replace('.wav', '') for file in test_files]

    for i in tqdm(range(len(test_feats)), desc="Testing"):
        mag_vocal_rec, mag_instr_rec = separate(test_feats[i], B_vocal, B_instr, num_iters)
        # Wiener Filter
        mag_instr_rec = (mag_instr_rec / (mag_vocal_rec + mag_instr_rec + 2.2204e-16)) * test_feats[i]
        mag_vocal_rec = (mag_vocal_rec / (mag_vocal_rec + mag_instr_rec + 2.2204e-16)) * test_feats[i]
        vocal_rec = librosa.istft(mag_vocal_rec * test_phases[i], hop_length=512, win_length=1024)
        instr_rec = librosa.istft(mag_instr_rec * test_phases[i], hop_length=512, win_length=1024)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        soundfile.write(f"{results_dir}/vocal_rec_{test_files[i]}.wav", vocal_rec, 16000)
        soundfile.write(f"{results_dir}/instr_rec_{test_files[i]}.wav", instr_rec, 16000)