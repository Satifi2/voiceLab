import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np


class AishellDataset(Dataset):
    def __init__(self, json_path, max_seq_length=460, feature_size=20):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())
        self.max_seq_length = max_seq_length
        self.feature_size = feature_size

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.data[key]
        encoder_input = torch.tensor(np.array(sample['encoder_input']), dtype=torch.float32)
        decoder_input = torch.tensor(np.array(sample['decoder_input']), dtype=torch.long)
        decoder_expected_output = torch.tensor(np.array(sample['decoder_expected_output']), dtype=torch.long)

        # 对 encoder_input 进行填充,  其实就是矩阵后面填上0向量
        if encoder_input.shape[0] < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - encoder_input.shape[0], self.feature_size)
            encoder_input = torch.cat([encoder_input, padding], dim=0)

        return key, encoder_input, decoder_input, decoder_expected_output

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                 max_seq_length, vocab_size):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.input_dim = input_dim
        self.embed_dim = embed_dim

    def forward(self, src, tgt):
        src = src + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        print("src.shape",src.shape)
        print("tgt.shape",tgt.shape)
        src=src.permute(1,0,2)
        tgt=tgt.permute(1,0,2)
        output = self.transformer(src, tgt)#transformer要求的张量形状是(seq_len, batch_size, embed_dim)
        output = self.fc_out(output)
        return output

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for keys, encoder_inputs, decoder_inputs, decoder_expected_output in dataloader:
            print(keys[0],'\n',encoder_inputs.shape,'\n',decoder_inputs.shape,'\n',decoder_expected_output.shape,'\n')
            encoder_inputs = encoder_inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_expected_output = decoder_expected_output.to(device)

            optimizer.zero_grad()
            decoder_outputs = model(encoder_inputs, decoder_inputs)
            #output的张量形状是(seq_len, batch_size, vocab_size)
            decoder_outputs = decoder_outputs.permute(1, 0, 2)
            print("decoder_outputs",decoder_outputs.shape)
            input_lengths = torch.full(size=(decoder_outputs.size(1),), fill_value=decoder_outputs.size(0), dtype=torch.long)
            target_lengths = torch.sum(decoder_expected_output != 0, dim=1)
            loss = criterion(decoder_outputs, decoder_expected_output,input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')

if __name__ == "__main__":
    # 超参数
    input_dim = 20
    embed_dim = 20
    num_heads = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 512
    max_seq_length = 460
    vocab_size = 4336
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    json_path = os.path.join(base_dir, 'S0002.json')
    dataset = AishellDataset(json_path)

    print('BAC009S0002W0122', dataset.data['BAC009S0002W0122']['decoder_input'])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                             max_seq_length, vocab_size)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, dataloader, criterion, optimizer, num_epochs, device)
