import numpy as np
import soundfile
import librosa
from hmmlearn import hmm
from dataset import dataset_train_hmm, dataset_test_hmm
import sys
from hmmlearn.base import ConvergenceMonitor

def train_hmm(train_mix_mfcc, n_states=2, n_mix=32, n_iter=10):

    model = hmm.GMMHMM(n_components=n_states, n_mix=n_mix, n_iter=n_iter, verbose=True)
    model.fit(train_mix_mfcc.T)
    
    return model

def map_states_to_labels(states, labels, n_states):
    state_label_map = {}
    for state in range(n_states):
        labels_unique, counts = np.unique(labels[states == state], return_counts=True)
        # normalize counts by the global distribution of labels
        global_distribution = np.bincount(labels, minlength=3) / len(labels)
        weighted_counts = counts / global_distribution[labels_unique]
        state_label_map[state] = labels_unique[np.argmax(weighted_counts)]
    return state_label_map

def test_hmm(model, test_mix_mfcc, state_label_map, auv_labels):
    auv_labels = auv_labels.astype(int)
    pred_states = model.predict(test_mix_mfcc.T)
    pred_labels = np.array([state_label_map[state] for state in pred_states])
    
    acc = np.sum(pred_labels == auv_labels) / len(auv_labels)
    print(f"Accuracy: {acc}")

    # Calculate accuracy for each label separately
    unique_labels = np.unique(auv_labels)
    label_accuracies = {}
    for label in unique_labels:
        label_indices = (auv_labels == label)
        label_acc = np.sum(pred_labels[label_indices] == auv_labels[label_indices]) / np.sum(label_indices)
        label_accuracies[label] = label_acc
        print(f"Accuracy for label {label}: {label_acc:.2f}")

    return pred_labels, acc

if __name__ == "__main__":
    train_mix_mfcc, auv_labels = dataset_train_hmm()
    # normalize the features
    # although axis=0 is wield, but performance better than axis=1
    train_mix_mfcc = (train_mix_mfcc - np.mean(train_mix_mfcc, axis=0, keepdims=True)) / np.std(train_mix_mfcc, axis=0, keepdims=True)
    model = train_hmm(train_mix_mfcc, n_iter=100)
    train_pred_state = model.predict(train_mix_mfcc.T)
    state_label_map = map_states_to_labels(train_pred_state, auv_labels, model.n_components)

    test_mix_mfcc, auv_labels_test = dataset_test_hmm()
    test_mix_mfcc = (test_mix_mfcc - np.mean(test_mix_mfcc, axis=0, keepdims=True)) / np.std(test_mix_mfcc, axis=0, keepdims=True)
    pred_labels, acc = test_hmm(model, test_mix_mfcc, state_label_map, auv_labels_test)
