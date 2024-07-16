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
        self.encoder_pos = utils.PositionalEncoding(encoder_seqlen, d_model).to(config.device)
        self.encoder_norm = nn.LayerNorm(d_model).to(config.device)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0).to(config.device)
        self.decoder_pos = utils.PositionalEncoding(decoder_seqlen, d_model).to(config.device)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True
        ).to(config.device)
        self.fc_out = nn.Linear(d_model, vocab_size).to(config.device)
    
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

    def predict(self, encoder_input, decoder_input, reverse_vocab):
        with torch.no_grad():
            output = self.forward(encoder_input, decoder_input)
            predicted_indices = torch.argmax(output, dim=-1)
            batch_size, seq_len = predicted_indices.size()
            predicted_words = []
            for i in range(batch_size):
                predicted_words.append([reverse_vocab[str(idx.item())] for idx in predicted_indices[i]])

            return predicted_indices, predicted_words

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
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for npz_file in npz_files:
            dataset = ASRDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
            dataset_loss = 0
            for batch in dataloader:
                wav_filenames, encoder_input, decoder_input, decoder_expected_output = batch
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
                dataset_loss += loss.item()
            print(f"Epoch {epoch + 1},file:{npz_file}, Loss: {total_loss / len(npz_files)}, data_set_loss:{dataset_loss}")
        if (epoch+1) % 5 ==0:
            utils.save_model_and_config(model, epoch, config.model_name)
    