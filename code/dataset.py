# Process MIR-1K dataset
import os
import numpy as np
import soundfile 
import librosa
import yaml

# set numpy random seed
np.random.seed(0)

males = [
    'abjones',  'bobon',        'bug',      'davidson', 
    'fdps',     'geniusturtle', 'jmzen',    'Kenshin', 
    'khair',    'leon',         'stool'
]
males_train = [
    'abjones',  'bobon',        'bug',      'davidson', 
    'fdps',     'geniusturtle', 'jmzen',    'Kenshin'
]
males_test = [
    'khair',    'leon',         'stool'
]

females = [
    'amy',      'Ani',          'annar',    'ariel', 
    'heycat',   'tammy',        'titon',    'yifen'
]
females_train = [
    'amy',      'Ani',          'annar',    'ariel', 
    'heycat',   'tammy',
]
females_test = [
    'titon',    'yifen'
]

names_train = males_train + females_train
names_test = males_test + females_test

# Load configuration from config.yaml
with open('./code/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

dataset_dir = config['dataset_dir']
wav_dir = config['wav_dir']

def split_dataset(wav_dir):
    # split the dataset into training and testing, 

    files = sorted(os.listdir(wav_dir))
    train_files = []
    test_files = []
    for file in files:
        file_name_split = file.replace('.wav', '').split('_')
        singer = file_name_split[0]
        song = file_name_split[1]
        part = file_name_split[2]
        if singer in names_train:
            train_files.append(file) 
        elif singer in names_test:
            test_files.append(file)
    
    # shuffle the files
    np.random.shuffle(train_files)
    np.random.shuffle(test_files)

    # save the file names into a file
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    with open(os.path.join(dataset_dir, 'train_files.txt'), 'w') as f:
        for file in train_files:
            f.write(os.path.join(wav_dir, file) + '\n')
    
    with open(os.path.join(dataset_dir, 'test_files.txt'), 'w') as f:
        for file in test_files:
            f.write(os.path.join(wav_dir, file) + '\n')


def merge_train_data(train_files_txt):
    with open(train_files_txt, 'r') as f:
        files = f.readlines()
    files = [file.strip() for file in files]
    instrs = []
    vocals = []

    for file in files:
        y, sr = soundfile.read(file)
        instr = y[:, 0] # instrumental
        vocal = y[:, 1] # vocal
        instrs.append(instr)
        vocals.append(vocal)
    
    merged_instrs = np.hstack(instrs)
    merged_vocals = np.hstack(vocals)

    soundfile.write(os.path.join(dataset_dir, 'train_instr.wav'), merged_instrs, sr)
    soundfile.write(os.path.join(dataset_dir, 'train_vocal.wav'), merged_vocals, sr)

def merge_data(files_txt, typ='test'):
    with open(files_txt, 'r') as f:
        files = f.readlines()
    files = [os.path.basename(file.strip()).replace('.wav', '') for file in files]
    instrs = []
    vocals = []
    merges_mfcc = []
    auv_label_merge = []

    for file in files:
        y, sr = soundfile.read(os.path.join(dataset_dir, 'MIR-1k', 'Wavfile', f"{file}.wav"))

        instr = y[:, 0] # instrumental
        vocal = y[:, 1] # vocal

        unvoiced_labels = np.loadtxt(os.path.join(dataset_dir, 'MIR-1k', 'UnvoicedFrameLabel', f"{file}.unv"))
        vocal_labels = np.loadtxt(os.path.join(dataset_dir, 'MIR-1k', 'vocal-nonvocalLabel', f"{file}.vocal"))
        auv_labels = [get_auv_label(unvoiced_label, vocal_label) for unvoiced_label, vocal_label in zip(unvoiced_labels, vocal_labels)]

        len_auv_frames = len(auv_labels)
    
        merge = instr + vocal
        merge = merge / np.max(np.abs(merge))
        merge_mfcc = extract_mfcc_features(merge, sr)

        len_merge_mfcc = merge_mfcc.shape[1]

        if len_auv_frames < len_merge_mfcc:
            # pad 0 to auv_labels
            merge_mfcc = merge_mfcc[:, :len_auv_frames]
        elif len_auv_frames > len_merge_mfcc:
            # truncate auv_labels
            auv_labels = auv_labels[:len_merge_mfcc]
        
        auv_label_merge.extend(auv_labels)
        
        instrs.append(instr)
        vocals.append(vocal)
        merges_mfcc.append(merge_mfcc)
    
    merged_instrs = np.hstack(instrs)
    merged_vocals = np.hstack(vocals)
    soundfile.write(os.path.join(dataset_dir, f"{typ}_instr.wav"), merged_instrs, sr)
    soundfile.write(os.path.join(dataset_dir, f"{typ}_vocal.wav"), merged_vocals, sr)

    auv_label_merge = np.array(auv_label_merge, dtype=int)
    np.savetxt(os.path.join(dataset_dir, f"{typ}_auv_label.txt"), auv_label_merge)

    merges_mfcc = np.concatenate(merges_mfcc, axis=1)
    np.save(os.path.join(dataset_dir, f"{typ}_mfcc.npy"), merges_mfcc)

def get_auv_label(unvoiced_label, vocal_label):
    if unvoiced_label == 5 and vocal_label == 0:
        return 0  # Accompaniment
    elif unvoiced_label == 5 and vocal_label == 1:
        # return 2  # Voiced
        return 1 # make all vocal frames (voiced and unvoiced) as vocal
    elif unvoiced_label in [1, 2, 3, 4] and vocal_label == 1:
        return 1  # Unvoiced
    else:
        return 0  # Default to Accompaniment

def merge_auv_label(train_files_txt):
    with open(train_files_txt, 'r') as f:
        files = f.readlines()
    files = [os.path.basename(file.strip()).replace('.wav', '') for file in files]

    auv_label_merge = []
    
    for file in files:
        unvoiced_labels = np.loadtxt(os.path.join(dataset_dir, 'MIR-1k', 'UnvoicedFrameLabel', f"{file}.unv"))
        vocal_labels = np.loadtxt(os.path.join(dataset_dir, 'MIR-1k', 'vocal-nonvocalLabel', f"{file}.vocal"))

        auv_labels = [get_auv_label(unvoiced_label, vocal_label) for unvoiced_label, vocal_label in zip(unvoiced_labels, vocal_labels)]
        auv_label_merge.extend(auv_labels)

    auv_label_merge = np.array(auv_label_merge)
    np.savetxt(os.path.join(dataset_dir, 'auv_label_merge.txt'), auv_label_merge)
    
    return auv_label_merge

def extract_feat(wav_file):
    y, sr = soundfile.read(wav_file)
    spec = librosa.stft(y, n_fft=1024, hop_length=512, win_length=1024)
    mag = abs(spec)
    phase = spec / (mag + 2.2204e-16)
    return mag, phase

def extract_mfcc(y, sr, n_mfcc=39):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=int(0.040 * sr), hop_length=int(0.020 * sr), win_length=int(0.040 * sr))
    return mfcc

def extract_mfcc_features(y, sr, n_mfcc=12):
    n_fft = int(sr * 0.04)
    frame_length = n_fft
    hop_length = int(sr * 0.02)
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=y, 
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Add energy
    energy = librosa.feature.rms(
        y=y, 
        frame_length=frame_length,
        hop_length=hop_length
    )
    log_energy = np.log(energy + 1e-10)
    
    # Delta features
    delta_mfcc = librosa.feature.delta(mfcc, width=3)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=3)
    delta_energy = librosa.feature.delta(log_energy, width=3)
    delta2_energy = librosa.feature.delta(log_energy, order=2, width=3)
    
    # Stack features
    features = np.vstack([
        mfcc, 
        log_energy,
        delta_mfcc, 
        delta_energy,
        delta2_mfcc,
        delta2_energy
    ])
    
    return features

def remix_extract_feat(wav_file):
    y, sr = soundfile.read(wav_file)
    instr = y[:, 0]
    vocal = y[:, 1]
    remix = instr + vocal
    spec = librosa.stft(remix, n_fft=1024, hop_length=512, win_length=1024)
    mag = abs(spec)
    phase = spec / (mag + 2.2204e-16)
    return mag, phase

def dataset_train():
    if not os.path.exists(dataset_dir) or not os.path.exists(os.path.join(dataset_dir, 'train_files.txt')) or not os.path.exists(os.path.join(dataset_dir, 'test_files.txt')):
        split_dataset(wav_dir)
    if not os.path.exists(os.path.join(dataset_dir, 'train_instr.wav')) or not os.path.exists(os.path.join(dataset_dir, 'train_vocal.wav')) or not os.path.exists(os.path.join(dataset_dir, 'train_vocal.wav')):
        merge_train_data(os.path.join(dataset_dir, 'train_files.txt'))

    # extract features
    train_instr_feat, train_instr_phase = extract_feat(os.path.join(dataset_dir, 'train_instr.wav'))
    train_vocal_feat, train_vocal_phase = extract_feat(os.path.join(dataset_dir, 'train_vocal.wav'))

    return  train_instr_feat, train_vocal_feat

def dataset_train_hmm():
    if not os.path.exists(dataset_dir) or not os.path.exists(os.path.join(dataset_dir, 'train_files.txt')) or not os.path.exists(os.path.join(dataset_dir, 'test_files.txt')):
        split_dataset(wav_dir)
    if not os.path.exists(os.path.join(dataset_dir, 'train_instr.wav')) or not os.path.exists(os.path.join(dataset_dir, 'train_vocal.wav')) or not os.path.exists(os.path.join(dataset_dir, 'train_mfcc.npy')):
        merge_data(os.path.join(dataset_dir, 'train_files.txt'), typ="train")
    
    auv_labels = np.loadtxt(os.path.join(dataset_dir, 'train_auv_label.txt'))
    auv_labels = auv_labels.astype(int)

    train_mix_mfcc = np.load(os.path.join(dataset_dir, 'train_mfcc.npy'), allow_pickle=True)

    assert train_mix_mfcc.shape[1] == len(auv_labels)
    return train_mix_mfcc, auv_labels # (n_mfcc, T)

def dataset_test():
    test_feats = []
    test_phases = []

    with open(os.path.join(dataset_dir, 'test_files.txt'), 'r') as f:
        files = f.readlines()
    files = [file.strip() for file in files]

    for file in files:
        feat, phase = remix_extract_feat(file)
        test_feats.append(feat)
        test_phases.append(phase)
    
    return test_feats, test_phases, files

def dataset_test_hmm():
    if not os.path.exists(dataset_dir) or not os.path.exists(os.path.join(dataset_dir, 'test_files.txt')) or not os.path.exists(os.path.join(dataset_dir, 'test_files.txt')):
        split_dataset(wav_dir)
    if not os.path.exists(os.path.join(dataset_dir, 'test_instr.wav')) or not os.path.exists(os.path.join(dataset_dir, 'test_vocal.wav')) or not os.path.exists(os.path.join(dataset_dir, 'test_mfcc.npy')):
        merge_data(os.path.join(dataset_dir, 'test_files.txt'), typ="test")
    
    auv_labels = np.loadtxt(os.path.join(dataset_dir, 'test_auv_label.txt'))
    auv_labels = auv_labels.astype(int)

    test_mix_mfcc = np.load(os.path.join(dataset_dir, 'test_mfcc.npy'), allow_pickle=True)

    assert test_mix_mfcc.shape[1] == len(auv_labels)
    return test_mix_mfcc, auv_labels # (n_mfcc, T)

