import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils

class BetterDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.wav_filenames = data['wav_filenames']
        self.source = data['encoder_input']
        self.target = data['decoder_input']
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        source = torch.tensor(self.source[idx], dtype=torch.float32)
        source_invalid = torch.all(source==0, dim=1)
        target = torch.tensor(self.target[idx], dtype=torch.long)[1:]
        target_invalid = target==0
        
        return self.wav_filenames[idx], source, target, source_invalid, target_invalid

def test_asr_dataset(dataloader):    
    for wav_filenames, source, target, source_invalid, target_invalid in dataloader:
        source_lengths, target_lengths = torch.sum(source_invalid == 0, dim=1), torch.sum(target_invalid == 0, dim=1)
        print("WAV Filenames:", wav_filenames[:3])
        print("Encoder Input:", source[0],source.shape)
        print("Decoder Input:", target[0],target.shape)
        print("source_invalid:", source_invalid[0])
        print("target_invalid:", target_invalid[0])
        print("source_lengths", source_lengths)
        print("target_lengths", target_lengths)
        break

if __name__ == '__main__':
    npz_file_path = output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
    dataset = BetterDataset(npz_file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    test_asr_dataset(dataloader)
    
