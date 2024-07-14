import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import config
from ASRDataset import *

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# 定义Transformer模型
class TransformerASR(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_len):
        super(TransformerASR, self).__init__()
        self.encoder_embedding = nn.Linear(d_model, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, encoder_input, decoder_input):
        encoder_input = self.encoder_embedding(encoder_input)
        encoder_input = self.pos_encoder(encoder_input)
        
        decoder_input = self.decoder_embedding(decoder_input)
        decoder_input = self.pos_encoder(decoder_input)
        
        transformer_output = self.transformer(
            encoder_input.permute(1, 0, 2),  # (seq_len, batch_size, d_model)
            decoder_input.permute(1, 0, 2)   # (seq_len, batch_size, d_model)
        )
        
        output = self.fc_out(transformer_output)
        return output.permute(1, 0, 2)  # (batch_size, seq_len, vocab_size)

# 加载数据和训练模型
if __name__ == '__main__':
    npz_file_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
    dataset = ASRDataset(npz_file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # test_asr_dataset(dataloader=dataloader)
    
    model = TransformerASR(
        vocab_size=config.vocab_size,
        d_model=config.mfcc_feature,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        max_seq_len=config.max_mfcc_seqlen
    )
    model = model.to(config.device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for wav_filenames, encoder_input, decoder_input, decoder_expected_output in dataloader:
            encoder_input = encoder_input.to(config.device)
            decoder_input = decoder_input.to(config.device)
            decoder_expected_output = decoder_expected_output.to(config.device)
            
            optimizer.zero_grad()
            output = model(encoder_input, decoder_input)
            
            output = output.reshape(-1, output.shape[-1])
            decoder_expected_output = decoder_expected_output.reshape(-1)
            loss = criterion(output, decoder_expected_output)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
