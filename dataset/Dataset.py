import torch
import torchaudio
import os

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.filepaths = []
        self.labels = []
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                self.filepaths.append(os.path.join(directory, filename))
                # Assuming the label is in the filename, e.g. "audio_file_0.wav" or "audio_file_1.wav"
                label = int(filename.split("_")[-1].split(".")[0])
                self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.filepaths[idx])
        if self.transform:
            audio = self.transform(audio)
        return audio, self.labels[idx]