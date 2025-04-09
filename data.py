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
from multiprocessing import Queue, Process
from queue import Empty

def get_fnames(folder):
    return sorted(glob.glob(f"{folder}/*.ogg"))

class BirdClefTrainAudio():
    """Loader for training data"""
    def __init__(self, data_dir, max_duration, sr=32000):
        self.taxonomy = pd.read_csv(f"{data_dir}/taxonomy.csv", index_col="primary_label")
        self.labels = self.taxonomy.index
        class_map = {class_name: i for i, class_name in enumerate(self.taxonomy.class_name.unique())}
        self.classes = {i: class_map[self.taxonomy.loc[label].class_name] for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)
        self.n_classes = self.taxonomy.class_name.nunique()
        self.max_duration = max_duration
        self.sr = sr
        self.data = [(fname, i)
                     for i, label in enumerate(self.labels)
                     for fname in get_fnames(f"{data_dir}/train_audio/{label}")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, label = self.data[idx]
        return fname, label

    def label_weights(self):
        freqs = Counter([label for _, label in self.data])
        return [self.n_labels/f**2 for f in freqs.values()]

    def class_weights(self):
        freqs = Counter(self.classes.values())
        return [self.n_labels/f for f in freqs.values()]

class BirdClefRawAudioData(Dataset):
    """Dataset for raw audio data"""
    def __init__(self, audio: BirdClefTrainAudio):
        self.audio = audio

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        fname, label = self.audio[idx]
        y, sr = librosa.load(fname, duration=self.audio.max_duration, sr=self.audio.sr)
        return torch.tensor(y.T, dtype=torch.float), torch.tensor(label, dtype=torch.long)

class BirdClefSTFTData(Dataset):
    """Dataset for training data"""
    def __init__(self, audio: BirdClefTrainAudio):
        self.audio = audio

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        fname, label = self.audio[idx]
        y, sr = librosa.load(fname, duration=self.audio.max_duration, sr=self.audio.sr)
        stft = np.abs(librosa.stft(y))
        return torch.tensor(stft.T, dtype=torch.float), torch.tensor(label, dtype=torch.long)

def process_harmonics(sr: int, max_duration: int, max_harmonic: int, fmin: int, fmax: int, in_queue: Queue, out_queue: Queue):
    hs = np.arange(1, max_harmonic)
    freqs = librosa.fft_frequencies(sr=sr)
    while True:
        try:
            next = in_queue.get(block=True, timeout=5)
        except Empty:
            return
        fname, label = next
        y, sr = librosa.load(fname, duration=max_duration, sr=sr)
        f0, v, vp = librosa.pyin(y=y, sr=sr, fmin=fmin, fmax=fmax, fill_na=0)
        s = np.abs(librosa.stft(y))
        harmonic_energy = librosa.f0_harmonics(s, f0=f0, harmonics=hs, freqs=freqs)
        out_queue.put((f0, harmonic_energy, label))

class BirdClefHarmonics(Dataset):
    """Dataset for the harmonics of the audio data"""
    def __init__(self, audio: BirdClefTrainAudio, fmin=50, fmax=4000, max_harmonic=11, num_processes=10):
        self.n = len(audio)
        in_queue=Queue(self.n)
        out_queue=Queue()
        for _ in range(num_processes):
            p = Process(target=process_harmonics, args=[audio.sr, audio.max_duration, max_harmonic, fmin, fmax, in_queue, out_queue])
            p.start()
        self.f0s = []
        self.harmonics = []
        self.labels = []
        for fname, label in audio:
            in_queue.put((fname, label))
        for i in range(self.n):
            f0, harmonic_energy, label = out_queue.get()
            self.f0s.append(f0)
            self.harmonics.append(harmonic_energy)
            self.labels.append(label)
            print(f"[{i:>5d}/{self.n:>5d}]", end="\r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        f0 = self.f0s[idx]
        harmonics = self.harmonics[idx]
        input = torch.tensor(np.concatenate([f0.reshape((1,-1)), harmonics], axis=0), dtype=torch.float).T
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input, label

class BirdClefClassDS(Dataset):
    def __init__(self, audio: BirdClefTrainAudio, base: Dataset):
        self.audio = audio
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, label = self.base[idx]
        return x, torch.tensor(self.audio.classes[label.item()], dtype=torch.long)
