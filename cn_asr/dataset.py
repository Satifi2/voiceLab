import os
import json
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config

def load_transcript(file_path):
    transcript_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                key = parts[0]
                value = ''.join(parts[1:])
                transcript_dict[key] = value
    return transcript_dict

def process_wav(wav_path, transcript_dict, vocab):
    file_id = os.path.splitext(os.path.basename(wav_path))[0]
    
    if file_id not in transcript_dict:
        return None, None
    
    transcript = transcript_dict[file_id]
    chars = list(transcript)
    
    char_ids = [vocab[char] for char in chars]
    
    audio, _ = librosa.load(wav_path, sr=config.sampling_rate)
    mfcc_matrix = librosa.feature.mfcc(y=audio, sr=config.sampling_rate, n_mfcc=config.mfcc_features).T
    # print("mfcc",mfcc_matrix.shape)
    
    return mfcc_matrix, char_ids

class AishellDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        transcript_path = '../data/data_aishell/transcript/aishell_transcript_v0.8.txt'
        self.transcript_dict = load_transcript(transcript_path)
        
        vocab_path = os.path.join('.', 'vocab.json')
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        self.data = []
        for wav_file in self.wav_files:
            wav_path = os.path.join(folder_path, wav_file)
            mfcc_matrix, char_ids = process_wav(wav_path, self.transcript_dict, self.vocab)
            if mfcc_matrix is not None and char_ids is not None:
                self.data.append((mfcc_matrix, char_ids, wav_file))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    dataset = AishellDataset('../data/data_aishell/wav/train/S0002')
    print(f"Total number of samples: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, (features, char_ids, wav_file) in enumerate(dataloader):
        print(f"Sample {i + 1}:{wav_file}")
        print(f"Feature shape: {features.shape}")
        print(f"Character IDs: {char_ids}")
        print()
        
        if i == 2:  # Print only the first 3 samples
            break
