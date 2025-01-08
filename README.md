# Karaoke Music Separation
This repository contains the implementation of an unsupervised framework for Blind Source Separation (BSS) in audio signals, specifically designed for separating vocal and accompaniment tracks in karaoke music. The proposed approach combines pitch-based binary masking and non-negative matrix factorization (NMF) to achieve high-quality separation without the need for labeled training data.

[[Introduction Video]](https://drive.google.com/file/d/15Mt1g5Qd_l55M_n-H6ARp4fXpuWETiz4/view?usp=sharing) [[Report]](./report.pdf)

**Key Features**

- **Unsupervised Framework**: No labeled data required, enabling flexible application to diverse and unlabeled audio environments.
- **Pitch-Based Binary Masking**: Extracts vocal harmonics by estimating the fundamental frequency of vocal signals, isolating vocal components in the spectrogram.
- **Non-Negative Matrix Factorization (NMF)**: Models and predicts accompaniment patterns using non-vocal spectrogram segments, allowing accurate separation of accompaniment from vocals.

**Separation Results**

<audio controls>
  <source src="results_final/abjones_1_03_instr_20241209-230219.wav" type="audio/mpeg">
</audio>

**Usage**

```
python code/separation.py
```

If you want to check the separation performance based only on NMF, which is our midterm-stage solution, please

```
python code/main.py
```

**Project Directory Structure**

*Code Files*

```
/code  
    ├── config.yaml         # Configuration file for dataset and result paths (Midterm solution)  
    ├── dataset.py          # Dataset processing for training and testing (Midterm solution)  
    ├── main.py             # Entry point to run the midterm stage code (Midterm solution)  
    ├── mfcc.py             # MFCC computation module (Midterm solution)  
    ├── nmf.py              # NMF model implementation (Used in both Midterm and Final solutions)  
    ├── pitch.py            # Pitch estimation using librosa (Final solution)  
    ├── pitch_hmm.py        # Pitch estimation using HMM model (Final solution)  
    ├── pitch_japan.py      # Pitch estimation via peak detection (Final solution)  
    └── separation.py       # Final-stage separation pipeline (Final solution)  
```

*Dataset*

```
/dataset  
    ├── Train and test data paths  
    ├── Processed train and test data  
    ├── MIR-1K dataset, please download at http://mirlab.org/dataset/public/
    └── Wave files to test the final solution
```

*Models*

```
/model  
    └── Trained bases for vocals and instrumentals (Midterm solution)  
```

*Results*

```
/results_midterm  
    └── Separated vocal and instrumental tracks of the test dataset (Midterm solution)  

/results_final  
    └── Separated vocal and instrumental tracks of `abjones_1_03.wav` from MIR-1K dataset (Final solution)  
```

*Assets*

```
/asset
	├── Binary mask expample of the final solution
	└── Spectrogram example of the separated vocal and instrumental tracks. 
```

*Report*

```
/report.pdf  
    └── Final project report  
```
