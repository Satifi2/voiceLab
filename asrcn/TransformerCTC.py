import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import config
from ASRDataset import *
import utils

class TransformerCTC(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, encoder_seqlen):
        super(TransformerCTC, self).__init__()
        self.encoder_pos = utils.PositionalEncoding(encoder_seqlen, d_model).to(config.device)
        self.encoder_norm = nn.LayerNorm(d_model).to(config.device)
        self.transformer_encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=config.dropout,activation='relu', batch_first=True),
        num_layers=num_encoder_layers).to(config.device)
        self.fc_out = nn.Linear(d_model, vocab_size).to(config.device)

    def forward(self, encoder_input):
        encoder_input = self.encoder_pos(encoder_input)
        encoder_input_pad = utils.pad_mask(encoder_input)
        transformer_output = self.transformer_encoder(encoder_input,src_key_padding_mask=encoder_input_pad)
        output = self.fc_out(transformer_output)
        return output

    def predict(self, encoder_input):
        with torch.no_grad():
            output = self.forward(encoder_input)
            output = nn.functional.log_softmax(output, dim=-1)
            predictions = []

            for i in range(encoder_input.size(0)):
                valid_output = output[i, :torch.sum(~utils.pad_mask(encoder_input)[i])]
                predictions.append(self.ctc_greedy_decoder(valid_output))

            return predictions

    def ctc_greedy_decoder(self, encoder_output):
        # print(encoder_output[0])
        max_probs_indices = encoder_output.argmax(dim=1)  
        decoded_sequence = []
        prev_token = None
        for t in range(max_probs_indices.shape[0]):
            token = max_probs_indices[t].item()
            if token != prev_token: 
                prev_token = token
                # if token != config.vocab_size-1:
                decoded_sequence.append(token)

        return decoded_sequence

if __name__ == '__main__':
    model = TransformerCTC(
        vocab_size=config.vocab_size,
        d_model=config.mfcc_feature,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        encoder_seqlen=config.max_mfcc_seqlen
    )
    model = model.to(config.device)
    criterion = nn.CTCLoss(blank=config.vocab_size-1, reduction="mean",zero_infinity=True)#blank=the last word id
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]

    num_epochs = 1
    num_datasets = 0
    save_dir = os.path.join('..','model','transformer_asr')
    utils.save_model_and_config(model, 999, config.model_name,save_dir)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for npz_file in npz_files:
            dataset = ASRDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
            dataset_loss = 0
            for batch in dataloader:
                wav_filenames, encoder_input, decoder_input, target_sequence = batch
                encoder_input = encoder_input.to(config.device)
                target_sequence = target_sequence[:,:-1].to(config.device)

                optimizer.zero_grad()
                output = model(encoder_input)
                print(encoder_input.shape,output.shape)#torch.Size([64, 460, 128]) torch.Size([64, 460, 4337])

                output = output.permute(1, 0, 2)  # [batch size, seq len, vocab size] -> [seq len, batch size, vocab size]
                output = nn.functional.log_softmax(output, dim=-1)

                input_lengths = torch.sum(~utils.pad_mask(encoder_input), dim=1)
                target_lengths = torch.sum(~utils.pad_mask(target_sequence), dim=1)

                # print(target_sequence[0],target_sequence[1])
                flattened_targets = []
                for i in range(target_sequence.size(0)):
                    flattened_targets.append(target_sequence[i, :target_lengths[i]])
                target_sequence = torch.cat(flattened_targets)
                # print(target_sequence)
                # print(input_lengths[0])

                loss = criterion(output, target_sequence, input_lengths, target_lengths)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                dataset_loss += loss.item()
            print(f"Epoch {epoch + 1}, file: {npz_file}, Loss: {total_loss / len(npz_files)}, dataset_loss: {dataset_loss}")
            num_datasets = num_datasets + 1
            if num_datasets == 30:
                utils.save_model_and_config(model, epoch + 1, config.model_name,save_dir)
