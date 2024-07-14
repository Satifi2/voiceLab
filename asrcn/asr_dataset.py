import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ASRDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.encoder_input = data['encoder_input']
        self.decoder_input = data['decoder_input']
        self.decoder_expected_output = data['decoder_expected_output']
    
    def __len__(self):
        print(len(self.encoder_input))
        return len(self.encoder_input)
    
    def __getitem__(self, idx):
        encoder_input = self.encoder_input[idx]
        decoder_input = self.decoder_input[idx]
        decoder_expected_output = self.decoder_expected_output[idx]
        
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        decoder_expected_output = torch.tensor(decoder_expected_output, dtype=torch.long)
        
        return encoder_input, decoder_input, decoder_expected_output

if __name__ == '__main__':
    npz_file_path = output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
    
    dataset = ASRDataset(npz_file_path)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    for encoder_input, decoder_input, decoder_expected_output in dataloader:
        print(encoder_input)
        print(decoder_input)
        print(decoder_expected_output)
        break
        print("encoder_input", encoder_input.shape)
        print("decoder_input", decoder_input.shape)
        print("decoder_expected_output", decoder_expected_output.shape)