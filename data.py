import glob
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import librosa

def get_fnames(folder):
    return sorted(glob.glob(f"{folder}/*.ogg"))

class BirdClefTrainAudio(Dataset):
    """Dataset for training data"""
    def __init__(self, data_dir):
        taxonomy = pd.read_csv(f"{data_dir}/taxonomy.csv")
        self.labels = taxonomy.primary_label.unique()
        self.n_labels = len(self.labels)
        self.data = [(fname, i)
                     for i, label in enumerate(self.labels)
                     for fname in get_fnames(f"{data_dir}/train_audio/{label}")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        y, sr = librosa.load(fname)
        stft = np.abs(librosa.stft(y))
        return stft, label
