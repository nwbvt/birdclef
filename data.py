import glob
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import librosa

def get_fnames(folder):
    return sorted(glob.glob(f"{folder}/*.ogg"))

class BirdClefTrainAudio(Dataset):
    """Dataset for training data"""
    def __init__(self, data_dir):
        taxonomy = pd.read_csv(f"{data_dir}/taxonomy.csv")
        self.data = [(fname, label)
                     for label in taxonomy.primary_label.unique()
                     for fname in get_fnames(f"{data_dir}/train_audio/{label}")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        y, sr = librosa.load(fname)
        stft = np.abs(librosa.stft(y))
        return stft, label
