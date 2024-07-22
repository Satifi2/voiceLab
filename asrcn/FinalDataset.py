import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils
from FinnalConfig import config

class BetterDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.wav_filenames = data['wav_filenames']
        self.source = data['encoder_input']
        self.target = data['decoder_input']
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        wav_filename= self.wav_filenames[idx]

        source = torch.tensor(self.source[idx], dtype=torch.float32)
        source_valid = torch.all(source!=config.pad_token, dim=1)
        source_length = torch.sum(source_valid)

        decoder_input = torch.tensor(self.target[idx])

        target = torch.tensor(self.target[idx])[1:]
        target = torch.cat([target, torch.tensor([config.pad_token])])
        eos_index = (target == config.pad_token).nonzero(as_tuple=True)[0][0]
        target[eos_index] = config.eos_token
        target_valid = target != config.pad_token
        target_length = torch.sum(target_valid)

        return wav_filename, source, decoder_input, target, source_valid, target_valid, source_length, target_length

def test_asr_dataset(dataloader):    
    for wav_filenames, source, decoder_input, target, source_valid, target_valid, source_lengths, target_lengths in dataloader:
        print("WAV Filenames:", wav_filenames[:3])
        print("Encoder Input:", source[0],source.shape)
        print("decoder_input", decoder_input[0], decoder_input.shape)
        print("Decoder Input:", target[0],target.shape)
        print("source_valid:", source_valid[0])
        print("target_valid:", target_valid[0])
        print("source_lengths", source_lengths)
        print("target_lengths", target_lengths)
        break

if __name__ == '__main__':
    npz_file_path = output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
    dataset = BetterDataset(npz_file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    test_asr_dataset(dataloader)
    
