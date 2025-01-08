import numpy as np
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
from hmmlearn import hmm
import os
from tqdm import tqdm
import time
import warnings
import soundfile as sf

class VoiceSeparationSystem:
	def __init__(self, sr=44100):
		"""Initialize VoiceSeparationSystem"""
		self.sr = sr
		self.frame_length = int(0.040 * self.sr)  # 40ms
		self.hop_length = int(0.020 * self.sr)    # 20ms
		
		# Add these parameters from the paper
		self.n_mfcc = 12  # Number of MFCC coefficients
		self.n_fft = self.frame_length  # Should be same as frame_length
		
		# Initialize HMM models
		self.auv_hmm = self._create_auv_hmm()
		self.pitch_hmm = self._create_pitch_hmm()

	def _create_auv_hmm(self):
		"""Create HMM for Accompaniment/Unvoiced/Voiced (A/U/V) decision"""
		model = hmm.GMMHMM(n_components=3, n_mix=32, covariance_type="diag")
		return model

	def _create_pitch_hmm(self):
		"""Create HMM for pitch tracking"""
		model = hmm.GMMHMM(n_components=36, n_mix=8, covariance_type="diag")
		return model

	def extract_mfcc_features(self, audio):
		"""Extract MFCC features"""
		print(f"Extracting MFCC features:")
		print(f"Input audio shape: {audio.shape}")
		
		# Extract MFCCs
		mfcc = librosa.feature.mfcc(
			y=audio, 
			sr=self.sr,
			n_mfcc=self.n_mfcc,
			hop_length=self.hop_length,
			n_fft=self.n_fft
		)
		print(f"Raw MFCC shape: {mfcc.shape}")
		
		# Add energy
		energy = librosa.feature.rms(
			y=audio, 
			frame_length=self.frame_length,
			hop_length=self.hop_length
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
		print(f"Final feature shape: {features.T.shape}")
		
		return features.T

	# Helper function to convert Hz to FFT bin index
	def hz_to_fft_bin(freq, sr, n_fft):
		return int(freq * n_fft / sr)

	def compute_f0_salience(self, audio, fmin=librosa.note_to_hz('D2'), 
						fmax=librosa.note_to_hz('D6')):
		"""Compute F0 salience function with 40ms frame length and 20ms hop length"""
		frame_length = int(0.040 * self.sr)  # 40ms
		hop_length = int(0.020 * self.sr)    # 20ms
		
		# Compute magnitude spectrogram
		D = librosa.stft(
			audio,
			n_fft=frame_length,
			hop_length=hop_length,
			win_length=frame_length
		)
		mag_spec = np.abs(D)
		
		# Whiten spectrum
		mag_spec_white = librosa.decompose.nn_filter(mag_spec)
		
		# Generate frequency grid
		freqs = librosa.fft_frequencies(sr=self.sr, n_fft=frame_length)
		
		# Initialize salience function
		times = librosa.frames_to_time(
			np.arange(mag_spec.shape[1]), 
			sr=self.sr, 
			hop_length=hop_length
		)
		salience = np.zeros((len(times), 361))
		
		# Parameters from paper
		K = 20  # Number of harmonics
		alpha = 27  # Hz
		beta = 320  # Hz

		def hz_to_fft_bin(freq, sr, n_fft):
			return int(freq * n_fft / sr)
		
		# Compute salience for each potential F0
		for i, f0 in enumerate(np.linspace(fmin, fmax, 361)):
			harmonic_sum = np.zeros(len(times))
			for k in range(1, K+1):
				freq = k * f0 + alpha
				if freq + beta < self.sr/2:  # Check Nyquist
					# Convert frequency to bin index
					freq_idx = hz_to_fft_bin(freq, self.sr, frame_length)
					if freq_idx < mag_spec_white.shape[0]:  # Check if bin is within range
						harmonic_sum += mag_spec_white[freq_idx, :]
			salience[:, i] = harmonic_sum
				
		return salience

	def extract_esi_features(self, salience):
		"""Extract Energy at Semitones of Interest (ESI) features"""
		esi_features = np.zeros((salience.shape[0], 36))  # 36 features for midi 39-74
		
		for n in range(36):
			midi_num = n + 39
			# Create triangle window
			window = self._create_triangle_window(midi_num)
			# Apply window and integrate
			esi_features[:, n] = np.convolve(salience[:, midi_num-39], window, mode='same')
			
		return esi_features

	def _create_triangle_window(self, center, width=2):
		"""Create triangle window centered at given midi number"""
		x = np.linspace(center - width, center + width, width*2)
		window = 1 - np.abs(x - center)
		window[window < 0] = 0
		return window

	def _init_nmf_matrices(self, V, n_components_voice=50, n_components_music=50):
		"""Initialize NMF matrices as described in Section 2.2"""
		F, T = V.shape
		
		# Initialize source-filter model matrices for voice
		BF = self._create_source_matrix()  # Fixed source matrix from KLGLOTT88
		AF = np.random.rand(BF.shape[1], T)  # Source activities
		BK = np.random.rand(F, n_components_voice)  # Filter shapes
		AK = np.random.rand(n_components_voice, T)  # Filter activities
		
		# Initialize accompaniment matrices
		BM = np.random.rand(F, n_components_music)  # Music bases
		AM = np.random.rand(n_components_music, T)  # Music activities
		
		return BF, AF, BK, AK, BM, AM

	def _create_source_matrix(self):
		"""Create source matrix BF using KLGLOTT88 model"""
		# Generate fundamental frequencies (f0s) for midi numbers 38.5 to 74.5
		midi_nums = np.arange(38.5, 74.6, 0.1)
		f0s = 440 * 2**((midi_nums - 69) / 12)
		
		# Initialize source matrix
		n_freq_bins = self.n_fft // 2 + 1
		BF = np.zeros((n_freq_bins, len(f0s)))
		
		# Generate glottal source spectra using KLGLOTT88 model
		for i, f0 in enumerate(f0s):
			spectrum = self._klglott88_spectrum(f0)
			BF[:, i] = spectrum[:n_freq_bins]
			
		return BF

	def _klglott88_spectrum(self, f0):
		"""Generate glottal source spectrum using KLGLOTT88 model"""
		duration = 1.0
		t = np.linspace(0, duration, int(self.sr * duration))
		
		# Generate one period of glottal pulse
		period = int(self.sr / f0)
		pulse = np.zeros(period)
		open_phase = int(0.6 * period)  # Open quotient = 0.6
		pulse[:open_phase] = 0.5 * (1 - np.cos(2 * np.pi * np.arange(open_phase) / open_phase))
		
		# Generate full signal and get spectrum
		signal = np.tile(pulse, int(np.ceil(len(t) / period)))[:len(t)]
		spectrum = np.abs(np.fft.fft(signal))
		
		return spectrum

	def _update_nmf_params(self, V, BF, AF, BK, AK, BM, AM):
		"""Update NMF parameters using multiplicative update rules"""
		eps = 1e-10  # Small constant to avoid division by zero

		def normalize_matrix(M):
			return M / (np.max(M) + eps)
		
		# Compute current model reconstruction
		DV = (BF @ AF) * (BK @ AK)  # Voice model
		DM = BM @ AM  # Music model
		D = DV + DM  # Total model
		
		# Update voice parameters
		# Update AF
		num = BF.T @ (V / (D + eps) * (BK @ AK))
		den = BF.T @ (np.ones_like(V) * (BK @ AK))
		AF = normalize_matrix(AF * num / (den + eps))
		
		# Update BK
		num = V / (D + eps) * (BF @ AF) @ AK.T
		den = np.ones_like(V) * (BF @ AF) @ AK.T
		BK = normalize_matrix(BK * num / (den + eps))
		
		# Update AK
		num = BK.T @ (V / (D + eps) * (BF @ AF))
		den = BK.T @ (np.ones_like(V) * (BF @ AF))
		AK *= num / (den + eps)
		
		# Update music parameters
		# Update BM
		num = V / (D + eps) @ AM.T
		den = np.ones_like(V) @ AM.T
		BM *= num / (den + eps)
		
		# Update AM
		num = BM.T @ (V / (D + eps))
		den = BM.T @ np.ones_like(V)
		AM *= num / (den + eps)
		
		return BF, AF, BK, AK, BM, AM

	def apply_soft_masking(self, mixed_audio, pitch_contour):
		eps = 1e-10
		
		# Compute STFT
		D = librosa.stft(mixed_audio, n_fft=self.n_fft, hop_length=self.hop_length)
		V = np.abs(D)**2
		
		# Normalize power spectrogram
		V = V / (np.max(V) + eps)
		
		# Initialize NMF matrices
		BF, AF, BK, AK, BM, AM = self._init_nmf_matrices(V)
		
		# First pass: Initialize voice and music components
		print("First NMF pass:")
		for i in range(50):
			# Update with normalization after each iteration
			BF, AF, BK, AK, BM, AM = self._update_nmf_params(V, BF, AF, BK, AK, BM, AM)
			
			if i % 10 == 0:
				DV = (BF @ AF) * (BK @ AK)
				DM = BM @ AM
				ratio = np.sum(DV) / (np.sum(DM) + eps)
				print(f"Iteration {i}: V/M ratio = {ratio:.4f}")
				
				# Normalize components to maintain balance
				if ratio > 2.0:  # If voice is too dominant
					AF *= np.sqrt(1.0 / ratio)
				elif ratio < 0.5:  # If music is too dominant
					AM *= np.sqrt(ratio)
		
		# Constrain AF based on pitch contour and voiced/unvoiced decision
		print("\nApplying pitch constraints:")
		AF_constrained = np.zeros_like(AF)
		n_modified = 0
		
		for t in range(AF.shape[1]):
			pitch = pitch_contour[t]
			if 39 <= pitch <= 74:  # Valid pitch range
				midi_idx = int((pitch - 38.5) * 10)
				# Use wider window for more natural sound
				window_size = 10
				idx_range = slice(
					max(0, midi_idx - window_size),
					min(AF.shape[0], midi_idx + window_size + 1)
				)
				AF_constrained[idx_range, t] = AF[idx_range, t]
				n_modified += 1
		
		print(f"Modified {n_modified}/{AF.shape[1]} frames based on pitch")
		
		# Second pass: Refine with constrained voice component
		print("\nSecond NMF pass:")
		for i in range(50):
			BF, AF_constrained, BK, AK, BM, AM = self._update_nmf_params(
				V, BF, AF_constrained, BK, AK, BM, AM
			)
			
			if i % 10 == 0:
				DV = (BF @ AF_constrained) * (BK @ AK)
				DM = BM @ AM
				ratio = np.sum(DV) / (np.sum(DM) + eps)
				print(f"Iteration {i}: V/M ratio = {ratio:.4f}")
		
		# Compute final spectrograms
		DV = (BF @ AF_constrained) * (BK @ AK)
		DM = BM @ AM
		
		# Balance the components
		total_power = np.sum(V)
		voice_ratio = 0.4  # Target 40% voice, 60% accompaniment
		scale_v = np.sqrt((total_power * voice_ratio) / (np.sum(DV) + eps))
		scale_m = np.sqrt((total_power * (1 - voice_ratio)) / (np.sum(DM) + eps))
		
		DV *= scale_v
		DM *= scale_m
		
		# Compute masks with temperature parameter
		temperature = 1.0
		voice_mask = (DV / (DV + DM + eps)) ** (1/temperature)
		music_mask = (DM / (DV + DM + eps)) ** (1/temperature)
		
		# Apply masks and reconstruct
		voice_stft = D * voice_mask
		music_stft = D * music_mask
		
		voice = librosa.istft(voice_stft, hop_length=self.hop_length)
		music = librosa.istft(music_stft, hop_length=self.hop_length)
		
		return voice, music

	def separate_voice(self, mixed_audio):
		"""Main method to separate voice from accompaniment"""
		# 1. A/U/V Decision
		mfcc_features = self.extract_mfcc_features(mixed_audio)
		
		# HMM expects 2D array
		if mfcc_features.ndim == 3:
			# Reshape if needed (frames, features)
			mfcc_features = mfcc_features.reshape(-1, mfcc_features.shape[-1])
		
		# Predict A/U/V states
		auv_states = self.auv_hmm.predict(mfcc_features)
		print("\nAUV Decision Statistics:")
		unique, counts = np.unique(auv_states, return_counts=True)
		print(f"AUV distribution: {dict(zip(unique, counts))}")
		
		# 2. Pitch Tracking for voiced segments
		salience = self.compute_f0_salience(mixed_audio)
		esi_features = self.extract_esi_features(salience)
		
		# Only perform pitch tracking on voiced segments
		voiced_segments = (auv_states == 2)
		print(f"Number of voiced segments: {np.sum(voiced_segments)}")
		
		pitch_states = np.zeros_like(auv_states)
		if np.any(voiced_segments):
			voiced_features = esi_features[voiced_segments]
			voiced_pitch = self.pitch_hmm.predict(voiced_features)
			
			# Verify pitch values
			print(f"Raw pitch prediction range: [{np.min(voiced_pitch)}, {np.max(voiced_pitch)}]")
			
			# Convert to MIDI range (39-74)
			voiced_pitch = np.clip(voiced_pitch, 0, 35)  # Ensure within state range
			voiced_pitch = 39 + voiced_pitch  # Convert to MIDI numbers
			print(f"MIDI pitch range: [{np.min(voiced_pitch)}, {np.max(voiced_pitch)}]")
			
			pitch_states[voiced_segments] = voiced_pitch
		
		# Convert pitch states to actual pitch values
		pitch_contour = 39 + pitch_states
		
		# 3. NMF-based Soft Masking
		print("\nNMF Soft Masking:")
		voice, accompaniment = self.apply_soft_masking(mixed_audio, pitch_contour)
		
		# Print statistics
		print("\nSeparation Statistics:")
		print(f"Input audio shape: {mixed_audio.shape}")
		print(f"Mixed audio range: [{np.min(mixed_audio)}, {np.max(mixed_audio)}]")
		print(f"Voice output range: [{np.min(voice)}, {np.max(voice)}]")
		print(f"Accompaniment output range: [{np.min(accompaniment)}, {np.max(accompaniment)}]")
		
		# Energy analysis
		voice_energy = np.sum(voice**2)
		acc_energy = np.sum(accompaniment**2)
		mixed_energy = np.sum(mixed_audio**2)
		print("\nEnergy Distribution:")
		print(f"Voice energy: {voice_energy:.4f}")
		print(f"Accompaniment energy: {acc_energy:.4f}")
		print(f"Mixed signal energy: {mixed_energy:.4f}")
		
		return voice, accompaniment, pitch_contour, auv_states

class VoiceSeparationTrainer(VoiceSeparationSystem):
	def __init__(self, sr=44100):
		"""Initialize trainer"""
		super().__init__(sr=sr)
		
	def _convert_to_mono(self, stereo_audio):
		"""Convert stereo to mono by averaging channels"""
		if stereo_audio.ndim == 2:
			return np.mean(stereo_audio, axis=0)
		return stereo_audio
		
	def _get_auv_label(self, unvoiced_label, vocal_label):
		if unvoiced_label == 5 and vocal_label == 0:
			return 0  # Accompaniment
		elif unvoiced_label == 5 and vocal_label == 1:
			return 2  # Voiced
		elif unvoiced_label in [1, 2, 3, 4] and vocal_label == 1:
			return 1  # Unvoiced
		else:
			return 0  # Default to Accompaniment
	
	def _process_pitch_labels(self, pitch_values, auv_labels):
		"""Process pitch values from .pv files"""
		# print("Processing pitch values:")
		# print(f"Input pitch range: [{np.min(pitch_values)}, {np.max(pitch_values)}]")
		# print(f"Number of voiced frames: {np.sum(auv_labels == 2)}")
		# print(f"Number of non-zero pitch values: {np.sum(pitch_values > 0)}")
		
		# Initialize MIDI pitch array
		midi_pitch = np.full_like(pitch_values, 39.0, dtype=float)
		
		# Only process for voiced frames with valid pitch
		voiced_frames = (auv_labels == 2)
		valid_pitch = (pitch_values > 0)
		valid_frames = voiced_frames & valid_pitch
		
		if np.any(valid_frames):
			# Get the valid frequencies
			freq_hz = pitch_values[valid_frames]
			# print(f"Valid pitch values (Hz): {freq_hz[:10]}")
			
			# Convert Hz to MIDI note numbers - adjusting for the lower frequency range
			# Multiply frequency by 2 to shift up one octave
			midi_values = 69 + 12 * np.log2((freq_hz * 2) / 440.0)
			# print(f"Converted MIDI values before clipping: {midi_values[:10]}")
			
			# Clip to valid range (39-74 as per paper)
			midi_values = np.clip(midi_values, 39, 74)
			midi_pitch[valid_frames] = midi_values
			
			# print(f"Number of valid pitch frames: {np.sum(valid_frames)}")
			# print(f"Final MIDI pitch range: [{np.min(midi_values)}, {np.max(midi_values)}]")
			
			# Print distribution of MIDI values
			unique_values, counts = np.unique(midi_values, return_counts=True)
			# print("MIDI value distribution:")
			# for value, count in zip(unique_values, counts):
			# 	print(f"MIDI {value:.1f}: {count} frames")
		
		return midi_pitch
	
	def load_dataset(self, dataset_path):
		print("Loading dataset...")
		training_data = []
		labels = {
			'auv': [],
			'pitch': []
		}
		
		wav_dir = os.path.join(dataset_path, 'Wavfile')
		wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
		
		for wav_file in tqdm(wav_files, desc="Loading files"):
			try:
				base_name = wav_file[:-4]
				
				# Load audio
				audio_path = os.path.join(wav_dir, wav_file)
				stereo_audio, _ = librosa.load(audio_path, sr=self.sr, mono=False)
				mono_audio = self._convert_to_mono(stereo_audio)
				training_data.append(mono_audio)
				
				# Load labels
				unv_path = os.path.join(dataset_path, 'UnvoicedFrameLabel', f'{base_name}.unv')
				vocal_path = os.path.join(dataset_path, 'vocal-nonvocalLabel', f'{base_name}.vocal')
				pitch_path = os.path.join(dataset_path, 'PitchLabel', f'{base_name}.pv')
				
				# Load and convert labels
				unvoiced_labels = np.loadtxt(unv_path)
				vocal_labels = np.loadtxt(vocal_path)
				pitch_values = np.loadtxt(pitch_path)
				
				# Convert to A/U/V labels
				auv_labels = np.array([
					self._get_auv_label(unv, vocal) 
					for unv, vocal in zip(unvoiced_labels, vocal_labels)
				])
				
				# Process pitch values
				if len(pitch_values) > 0:  # Check if we have pitch values
					midi_pitch = self._process_pitch_labels(pitch_values, auv_labels)
					
					# Verify label lengths match
					n_frames = min(len(midi_pitch), len(auv_labels))
					labels['pitch'].append(midi_pitch[:n_frames])
					labels['auv'].append(auv_labels[:n_frames])
				
			except Exception as e:
				print(f"Error processing {wav_file}: {str(e)}")
				continue
			
			# Print statistics for first few files
			if len(labels['auv']) <= 5:
				print(f"\nFile {base_name} statistics:")
				uniq_auv, counts_auv = np.unique(labels['auv'][-1], return_counts=True)
				print(f"AUV labels: {dict(zip(uniq_auv, counts_auv))}")
				pitch = labels['pitch'][-1]
				voiced_pitch = pitch[labels['auv'][-1] == 2]
				if len(voiced_pitch) > 0:
					print(f"Pitch range in voiced segments: [{np.min(voiced_pitch)}, {np.max(voiced_pitch)}]")
		
		print(f"Successfully loaded {len(training_data)} files")
		return training_data, labels
	
	def extract_mfcc_features(self, audio):
		"""Modified MFCC feature extraction with 40ms frame length and 20ms hop length"""
		try:
			# print(f"Audio shape: {audio.shape}, Length in seconds: {len(audio)/self.sr:.2f}")
			
			# Calculate frame and hop lengths in samples
			frame_length = int(0.040 * self.sr)  # 40ms
			hop_length = int(0.020 * self.sr)    # 20ms
			
			# print(f"Frame length: {frame_length}, Hop length: {hop_length}")
			
			# Extract MFCCs with smaller delta window
			mfcc = librosa.feature.mfcc(
				y=audio, 
				sr=self.sr,
				n_mfcc=12,
				n_fft=frame_length,
				hop_length=hop_length,
				win_length=frame_length
			)
			# print(f"MFCC shape: {mfcc.shape}")
			
			# Compute energy with same frame and hop lengths
			energy = librosa.feature.rms(
				y=audio, 
				frame_length=frame_length,
				hop_length=hop_length
			)
			log_energy = np.log(energy + 1e-10)
			# print(f"Energy shape: {log_energy.shape}")
			
			# Calculate deltas with smaller width
			delta_mfcc = librosa.feature.delta(mfcc, width=3)
			delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=3)
			delta_energy = librosa.feature.delta(log_energy, width=3)
			delta2_energy = librosa.feature.delta(log_energy, order=2, width=3)
			
			# Combine features
			features = np.vstack([
				mfcc, 
				log_energy,
				delta_mfcc, 
				delta_energy,
				delta2_mfcc,
				delta2_energy
			])
			
			# print(f"Final features shape: {features.shape}")
			return features.T
			
		except Exception as e:
			print(f"Error in MFCC extraction: {str(e)}")
			print(f"Error details: {type(e).__name__}")
			import traceback
			print(traceback.format_exc())
			return None

	def prepare_features(self, training_data):
		print("Extracting features...")
		print(f"Number of audio clips to process: {len(training_data)}")
		
		mfcc_features = []
		esi_features = []
		
		for i, audio in enumerate(tqdm(training_data, desc="Extracting features")):
			try:
				# print(f"\nProcessing audio clip {i}")
				# Extract MFCC features
				mfcc = self.extract_mfcc_features(audio)
				if mfcc is not None:
					# print(f"Successfully extracted MFCC features for clip {i}, shape: {mfcc.shape}")
					
					# Extract ESI features
					salience = self.compute_f0_salience(audio)
					esi = self.extract_esi_features(salience)
					# print(f"Successfully extracted ESI features for clip {i}, shape: {esi.shape}")
					
					# Ensure features and labels have same number of frames
					n_frames = min(mfcc.shape[0], esi.shape[0])
					mfcc_features.append(mfcc[:n_frames])
					esi_features.append(esi[:n_frames])
					# print(f"After alignment - MFCC shape: {mfcc[:n_frames].shape}, ESI shape: {esi[:n_frames].shape}")
				else:
					print(f"Failed to extract features for audio clip {i}")
					
			except Exception as e:
				print(f"Error processing audio clip {i}: {str(e)}")
				print(f"Error type: {type(e).__name__}")
				import traceback
				print(traceback.format_exc())
				continue

		return mfcc_features, esi_features

	def train(self, dataset_path):
		start_time = time.time()
		
		print("=== Starting Training Process ===")
		training_data, labels = self.load_dataset(dataset_path)
		
		# Add debug prints for labels
		print("\n=== Label Statistics ===")
		print("Number of training files:", len(training_data))
		for i in range(min(5, len(labels['auv']))):  # Print first 5 files
			print(f"\nFile {i} statistics:")
			uniq_auv, counts_auv = np.unique(labels['auv'][i], return_counts=True)
			print(f"AUV labels distribution: {dict(zip(uniq_auv, counts_auv))}")
			uniq_pitch, counts_pitch = np.unique(labels['pitch'][i], return_counts=True)
			print(f"Pitch values range: [{np.min(labels['pitch'][i])}, {np.max(labels['pitch'][i])}]")
		
		if not training_data:
			raise ValueError("No training data was loaded")
			
		print("\n=== Feature Extraction ===")
		try:
			mfcc_features, esi_features = self.prepare_features(training_data)
			
			print("\n=== Training AUV Model ===")
			X_auv = []
			y_auv = []
			lengths_auv = []
			
			# Add debug prints for feature extraction
			print("\nProcessing features:")
			for i, (mfcc, auv_label) in enumerate(zip(mfcc_features, labels['auv'])):
				print(f"\nProcessing file {i}:")
				print(f"MFCC shape: {mfcc.shape}")
				print(f"AUV label shape: {auv_label.shape}")
				
				n_frames = min(len(mfcc), len(auv_label))
				X_auv.append(mfcc[:n_frames])
				y_auv.append(auv_label[:n_frames])
				lengths_auv.append(n_frames)
				
				if i < 5:  # Print details for first 5 files
					uniq, counts = np.unique(auv_label[:n_frames], return_counts=True)
					print(f"AUV labels in processed data: {dict(zip(uniq, counts))}")
			
			X_auv = np.vstack(X_auv)
			y_auv = np.concatenate(y_auv)
			print(f"\nFinal AUV training data shape: {X_auv.shape}")
			print(f"AUV labels distribution in training data:")
			uniq, counts = np.unique(y_auv, return_counts=True)
			print(dict(zip(uniq, counts)))
			print(f"Sequence lengths: {lengths_auv}")
			
			# Verify HMM model before training
			print("\nAUV HMM parameters before training:")
			print(f"n_components: {self.auv_hmm.n_components}")
			print(f"n_mix: {self.auv_hmm.n_mix}")
			
			self.auv_hmm.fit(X_auv, lengths=lengths_auv)
			
			print("\n=== Training Pitch Model ===")
			X_pitch = []
			y_pitch = []
			lengths_pitch = []
			
			for i, (esi, auv_label, pitch_label) in enumerate(zip(esi_features, labels['auv'], labels['pitch'])):
				# Ensure all arrays have same length
				n_frames = min(len(esi), len(auv_label), len(pitch_label))
				auv_label = auv_label[:n_frames]
				pitch_label = pitch_label[:n_frames]
				esi = esi[:n_frames]
				
				voiced_frames = (auv_label == 0)
				if np.any(voiced_frames):
					pitch_feat = esi[voiced_frames]
					X_pitch.append(pitch_feat)
					y_pitch.append(pitch_label[voiced_frames])
					lengths_pitch.append(len(pitch_feat))
			
			if not X_pitch:
				raise ValueError("No voiced frames found in the dataset")
				
			X_pitch = np.vstack(X_pitch)
			y_pitch = np.concatenate(y_pitch)
			print(f"Pitch training data shape: {X_pitch.shape}")
			print(f"Pitch sequence lengths: {lengths_pitch}")
			self.pitch_hmm.fit(X_pitch, lengths=lengths_pitch)
			
			total_time = time.time() - start_time
			print(f"\n=== Training Complete ===")
			print(f"Total training time: {total_time/60:.2f} minutes")
			
			return {
				'total_files': len(training_data),
				'successful_features': len(mfcc_features),
				'training_time': total_time,
				'auv_data_shape': X_auv.shape,
				'pitch_data_shape': X_pitch.shape
			}
			
		except Exception as e:
			print(f"Training failed: {str(e)}")
			raise

def train_models():
	# Initialize
	trainer = VoiceSeparationTrainer(sr=44100)
	
	# Train models
	dataset_path = "D:\\mlsp_dataset\\MIR-1K"
	stats = trainer.train(dataset_path)
	
	# Save trained models
	print("\nSaving models...")
	import joblib
	joblib.dump(trainer.auv_hmm, 'auv_model.pkl')
	joblib.dump(trainer.pitch_hmm, 'pitch_model.pkl')
	
	# Print training statistics
	print("\nTraining Statistics:")
	print(f"Total files processed: {stats['total_files']}")
	print(f"Total training time: {stats['training_time']/60:.2f} minutes")
	print(f"AUV data shape: {stats['auv_data_shape']}")
	print(f"Pitch data shape: {stats['pitch_data_shape']}")

train_models()

def separate_song(model_path, song_path):
    """
    Separate voice from a song using trained models
    
    Args:
        model_path: Path to directory containing trained models
        song_path: Path to the song to separate
    """
    # Load trained models
    import joblib
    
    # Create separator instance
    separator = VoiceSeparationSystem(sr=44100)
    
    # Load trained models
    separator.auv_hmm = joblib.load(os.path.join(model_path, 'auv_model.pkl'))
    separator.pitch_hmm = joblib.load(os.path.join(model_path, 'pitch_model.pkl'))
    
    # Load and process song
    audio, sr = librosa.load(song_path, sr=44100, mono=True)
    
    # Get voice and accompaniment
    voice, accompaniment, pitch_contour, auv_states = separator.separate_voice(audio)
    
    # Save results
    sf.write('separated_voice.wav', voice, sr)
    sf.write('separated_accompaniment.wav', accompaniment, sr)
    
    return voice, accompaniment, pitch_contour, auv_states

trained_model_path = "./"
song_path = "E:\\24_fall_backup\\midterm_model_n_code_1\\dataset\\abjones_1.wav"
separate_song(trained_model_path, song_path)

