import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from FinnalConfig import config


def create_padding_mask(lengths, max_len):
    return torch.arange(max_len).expand(len(lengths), max_len).to(config.device) >= lengths.unsqueeze(1)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class FinalDataset(Dataset):
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
        source_invalid = torch.all(source==config.pad_token, dim=1)
        source_length = torch.sum(source_invalid == False)

        decoder_input = torch.tensor(self.target[idx])

        target = torch.tensor(self.target[idx])[1:]
        target = torch.cat([target, torch.tensor([config.pad_token])])
        eos_index = (target == config.pad_token).nonzero(as_tuple=True)[0][0]
        target[eos_index] = config.eos_token
        target_invalid = target == config.pad_token
        target_length = torch.sum(target_invalid == False)

        return wav_filename, source.to(config.device), \
            decoder_input.to(config.device), target.to(config.device), \
            source_invalid.to(config.device), target_invalid.to(config.device), \
            source_length.to(config.device), target_length.to(config.device)


def test_dataset(dataloader):    
    for wav_filenames, source, decoder_input, target, source_valid, target_valid, source_lengths, target_lengths in dataloader:
        print("WAV Filenames:", wav_filenames[:3])
        print("Encoder Input:", source[0],source.shape)
        print("decoder_input", decoder_input[0], decoder_input.shape)
        print("Decoder Input:", target[0],target.shape)
        print("source_valid:", source_valid[0])
        print("target_valid:", target_valid[0])
        print("target_lengths to target mask",create_padding_mask(target_lengths, target.shape[1])[0])
        print("source_lengths", source_lengths)
        print("target_lengths", target_lengths)
        break


if __name__ == '__main__':
    npz_file_path = output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
    dataset = FinalDataset(npz_file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    test_dataset(dataloader)
    
