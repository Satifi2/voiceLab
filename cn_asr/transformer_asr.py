import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from dataset import AishellDataset
import config
import utils

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerASR(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerASR, self).__init__()
        
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)
        
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        
        output = self.transformer(src, tgt)
        return self.fc_out(output)

def train(model, dataset, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for i, (mfcc_matrix, char_ids, wav_file) in enumerate(dataset):
        mfcc_matrix = torch.tensor(mfcc_matrix).transpose(0, 1).float().to(device)  # (time, feature)
        char_ids = torch.tensor(char_ids).long().to(device)

        # Prepare input and output for the model
        decoder_input = torch.cat([torch.tensor([config.bos_token], device=device), char_ids])
        target = torch.cat([char_ids, torch.tensor([config.eos_token], device=device)])

        print(mfcc_matrix,mfcc_matrix.shape,decoder_input,decoder_input.shape,target)

        optimizer.zero_grad()
        output = model(mfcc_matrix, decoder_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Sample {i+1}, Loss: {loss.item():.4f}')

    return total_loss / len(dataset)

if __name__ == "__main__":
    # Load vocabulary
    with open('./vocab.json', 'r') as f:
        vocab = json.load(f)
    
    with open('./reverse_vocab.json', 'r') as f:
        reverse_vocab = json.load(f)

    # Set up the model
    model = TransformerASR(
        input_dim=config.mfcc_features,
        vocab_size=config.vocab_size,
        d_model=config.mfcc_features,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        max_seq_length=max(config.max_sentence_len, config.max_mfcc_seqlen),
        dropout=config.dropout
    ).to(config.device)

    # Set up the dataset
    dataset = AishellDataset('../data/data_aishell/wav/train/S0002')

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train for one epoch
    epoch_loss = train(model, dataset, criterion, optimizer, config.device)
    print(f'Epoch 1 completed. Average Loss: {epoch_loss:.4f}')

    # Save the model
    save_dir = os.path.join('..', 'model', 'crossentropy')
    utils.save_model_and_config(model, 1, config.model_name, save_dir)
    print(f'Model saved to {save_dir}')