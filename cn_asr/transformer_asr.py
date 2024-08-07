import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from dataset import AishellDataset
from config import config
import utils
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class TransformerASR(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerASR, self).__init__()
        
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = utils.PositionalEncoding(d_model)
        
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
        # print("src",src.shape)
        # print("tgt",tgt.shape)

        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)
        
        tgt = self.decoder_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        
        # print("src",src.shape)
        # print("tgt",tgt.shape)
        output = self.transformer(src, tgt)
        return self.fc_out(output)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, accumulated_loss, batch_count = 0, 0, 0
    batch_size = config.dataloader_batch_size

    for i, (mfcc_matrix, char_ids, wav_file) in enumerate(dataloader):
        mfcc_matrix = mfcc_matrix.squeeze(0).to(device)
        char_ids = torch.tensor(char_ids).long().to(device)
        # print("mfcc_matrix",mfcc_matrix,mfcc_matrix.shape)
        # print("char_ids",char_ids,char_ids.shape)

        decoder_input = torch.cat([torch.tensor([config.bos_token], device=device), char_ids])
        target = torch.cat([char_ids, torch.tensor([config.eos_token], device=device)])
        # print("decoder_input",decoder_input,decoder_input.shape)
        # print("target",target,target.shape)

        decoder_output = model(mfcc_matrix, decoder_input)
        loss = criterion(decoder_output, target)
        accumulated_loss += loss
        batch_count += 1

        if (i+1) % batch_size == 0 or (i+1) == len(dataloader) :
            average_loss = accumulated_loss / batch_count
            average_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss.item()
            accumulated_loss, batch_count = 0, 0
            
            print(f'Sample {i+1}, Loss: {loss.item():.4f}')            

    return total_loss / len(dataloader)

if __name__ == "__main__":    
    with open('./reverse_vocab.json', 'r') as f:
        reverse_vocab = json.load(f)

    model = TransformerASR(
        input_dim=config.mfcc_feature,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        dropout=config.dropout
    ).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dir = '../data/data_aishell/wav/train/'
    all_dirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir)]

    for epoch in range(1, 6):
        for directory in all_dirs:
            print(f"Training on directory: {directory}")
            dataset = AishellDataset(directory)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            train(model, dataloader, criterion, optimizer, config.device)
        save_dir = os.path.join('..', 'model', 'crossentropy')
        utils.save_model_and_config(model, epoch, config.model_name, save_dir)
        print(f'Model saved to {save_dir}')