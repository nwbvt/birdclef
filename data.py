import glob
import re
from collections import Counter
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import librosa
from mimir.training import DEVICE

def get_fnames(folder):
    return sorted(glob.glob(f"{folder}/*.ogg"))

class BirdClefTrainAudio(Dataset):
    """Dataset for training data"""
    def __init__(self, data_dir, max_duration):
        taxonomy = pd.read_csv(f"{data_dir}/taxonomy.csv")
        self.labels = taxonomy.primary_label.unique()
        self.n_labels = len(self.labels)
        self.max_duration = max_duration
        self.data = [(fname, i)
                     for i, label in enumerate(self.labels)
                     for fname in get_fnames(f"{data_dir}/train_audio/{label}")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        y, sr = librosa.load(fname, duration=self.max_duration)
        stft = np.abs(librosa.stft(y))
        return torch.tensor(stft.T, dtype=torch.float), torch.tensor(label, dtype=torch.long)

    def label_weights(self):
        freqs = Counter([label for _, label in self.data])
        return torch.tensor([1/f for f in freqs.values()]).to(DEVICE)
