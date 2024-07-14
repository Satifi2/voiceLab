import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
    for wav_filenames, encoder_input, decoder_input, decoder_expected_output in dataloader:
        # print(encoder_input)
        # print(decoder_input)
        # print(decoder_expected_output)
        # break
        print("wav_filenames",wav_filenames,f"共计有:{len(wav_filenames)}个")
        print("encoder_input", encoder_input.shape)
        print("decoder_input", decoder_input.shape)
        print("decoder_expected_output", decoder_expected_output.shape)

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
encoder_input torch.Size([32, 460, 128])
decoder_input torch.Size([32, 31])
decoder_expected_output torch.Size([32, 31])
encoder_input torch.Size([32, 460, 128])
...
'''

'''
        # print(decoder_input)
        # print(decoder_expected_output)
array([[  1, 140,   6, ...,   0,   0,   0],
       [  1,  13, 226, ...,   0,   0,   0],
       [  1, 139,  49, ...,   0,   0,   0],
       ...,
       [  1,  67,  71, ...,   0,   0,   0],
       [  1, 830, 918, ...,   0,   0,   0],
       [  1, 138, 124, ...,   0,   0,   0]]), 
       
array([[140,   6, 147, ...,   0,   0,   2],
       [ 13, 226, 227, ...,   0,   0,   2],
       [139,  49, 213, ...,   0,   0,   2],
       ...,
       [ 67,  71, 142, ...,   0,   0,   2],
       [830, 918,  72, ...,   0,   0,   2],
       [138, 124,  19, ...,   0,   0,   2]]))
'''