import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import utils

class ASRDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.wav_filenames = data['wav_filenames']
        self.encoder_input = data['encoder_input']
        self.decoder_input = data['decoder_input']
        self.decoder_expected_output = data['decoder_expected_output']
    
    def __len__(self):
        return len(self.encoder_input)
    
    def __getitem__(self, idx):
        encoder_input = self.encoder_input[idx]
        decoder_input = self.decoder_input[idx]
        decoder_expected_output = self.decoder_expected_output[idx]
        
        encoder_input = torch.tensor(encoder_input, dtype=torch.float32)
        decoder_input = torch.tensor(decoder_input, dtype=torch.long)
        decoder_expected_output = torch.tensor(decoder_expected_output, dtype=torch.long)
        
        return self.wav_filenames[idx], encoder_input, decoder_input, decoder_expected_output

def test_asr_dataset(dataloader):
    data_iter = iter(dataloader)
    
    wav_filenames, encoder_input, decoder_input, decoder_expected_output = next(data_iter)
    print("WAV Filenames:", wav_filenames[0])
    print("Encoder Input:", encoder_input.shape)
    print("Decoder Input:", decoder_input[0],decoder_input.shape)
    print("Decoder Expected Output:", decoder_expected_output[0],decoder_expected_output.shape)
    print("Encoder Input Mask",utils.pad_mask(encoder_input)[0],utils.pad_mask(encoder_input).shape)
    print("Decoder Input Mask",utils.pad_mask(decoder_input)[0],utils.pad_mask(decoder_input).shape)
    return encoder_input[0],decoder_input[0]

if __name__ == '__main__':
    npz_file_path = output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
    
    dataset = ASRDataset(npz_file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    test_asr_dataset(dataloader)
    


'''
364#wav文件总数
encoder_input torch.Size([32, 460, 128])
decoder_input torch.Size([32, 31])
decoder_expected_output torch.Size([32, 31])
...

array([[  1, 140,   6, ...,   0,   0,   0],
       [  1,  13, 226, ...,   0,   0,   0],
       [  1, 139,  49, ...,   0,   0,   0],
       ...,
       [  1,  67,  71, ...,   0,   0,   0],
       [  1, 830, 918, ...,   0,   0,   0],
       [  1, 138, 124, ...,   0,   0,   0]]), 
'''