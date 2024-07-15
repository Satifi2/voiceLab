import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import config
from ASRDataset import *
import utils

class TransformerASR(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, encoder_seqlen, decoder_seqlen):
        super(TransformerASR, self).__init__()
        self.encoder_pos = utils.PositionalEncoding(encoder_seqlen,d_model)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.decoder_pos = utils.PositionalEncoding(decoder_seqlen,d_model)
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
        encoder_input_pad = utils.pad_mask(encoder_input)
        decoder_input_pad = utils.pad_mask(decoder_input)

        # encoder_input = self.encoder_norm(encoder_input)
        encoder_input = self.encoder_pos(encoder_input)
        
        decoder_input = self.decoder_embedding(decoder_input)
        decoder_input = self.decoder_pos(decoder_input)
        
        transformer_output = self.transformer(
            src=encoder_input,
            tgt=decoder_input,
            src_key_padding_mask=encoder_input_pad,  
            tgt_key_padding_mask=decoder_input_pad, 
            memory_key_padding_mask=encoder_input_pad  
        )
        
        output = self.fc_out(transformer_output)
        return output

if __name__ == '__main__':
    model = TransformerASR(
        vocab_size=config.vocab_size,
        d_model=config.mfcc_feature,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        encoder_seqlen=config.max_mfcc_seqlen,
        decoder_seqlen=config.max_sentence_len
    )
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    npz_file_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
    dataset = ASRDataset(npz_file_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # test_asr_dataset(dataloader=dataloader)
    
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