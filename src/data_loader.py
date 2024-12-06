from glob import glob
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.files = self._load_files()

    def _load_files(self):
        files = []
        for subfolder in os.listdir(self.data_path):
            path = os.path.join(self.data_path, subfolder)
            files.append(
                (
                    list(glob(os.path.join(path, "mix_snr_*.wav")))[0],
                    os.path.join(path, 'voice.wav'),
                    os.path.join(path, 'noise.wav')
                )
            )
        return files
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mix_file, voice_file, noise_file = self.files[idx]

        mix_audio, _ = torchaudio.load(mix_file)
        voice_audio, _ = torchaudio.load(voice_file)
        noise_audio, _ = torchaudio.load(noise_file)

        mix_tensor = torch.FloatTensor(mix_audio / mix_audio.abs().max())
        voice_tensor = torch.FloatTensor(voice_audio / voice_audio.abs().max())
        noise_tensor = torch.FloatTensor(noise_audio / noise_audio.abs().max())

        return mix_tensor, voice_tensor, noise_tensor

def get_dataloader(data_path, batch_size=32, shuffle=True):
    dataset = AudioDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
