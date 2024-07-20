import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from dataset import AishellDataset
from config import config
import utils
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import log_softmax

class TransformerCTC(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=config.dropout):
        super(TransformerCTC, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=config.mfcc_feature,out_channels=d_model,padding=0,kernel_size=1,stride=1)
        self.encoder_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = utils.PositionalEncoding(d_model)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_encoder_layers
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src):
        # src = src.transpose(0,1)
        # src = self.conv1(src)
        # src = src.transpose(0,1)
        src = self.encoder_embedding(src)
        src = self.positional_encoding(src)
        src = self.encoder(src)
        src = self.fc_out(src)
        return src
    
    def greedy_decoder(self,output):
        print(output.argmax(dim=1)) 

def train(model, dataset, criterion, optimizer, device):
    model.train()
    total_loss, accumulated_loss, batch_count = 0, 0, 0
    batch_size = config.dataloader_batch_size

    for i, (mfcc_matrix, char_ids, wav_file) in enumerate(dataset):
        mfcc_matrix = mfcc_matrix.squeeze(0).to(device)
        char_ids = torch.tensor(char_ids).long().to(device)
        
        input_lengths = torch.tensor([mfcc_matrix.size(0)], dtype=torch.long).to(device)
        target_lengths = torch.tensor([char_ids.size(0)], dtype=torch.long).to(device)

        output = model(mfcc_matrix)
        output = log_softmax(output, dim=-1)
        
        # Compute CTC loss
        loss = criterion(output, char_ids, input_lengths, target_lengths)
        accumulated_loss += loss
        batch_count += 1

        if (i+1) % batch_size == 0 or (i+1) == len(dataset):
            model.greedy_decoder(output)
            average_loss = accumulated_loss / batch_count
            average_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += accumulated_loss.item()
            accumulated_loss, batch_count = 0, 0
            
            print(f'Sample {i+1}, Loss: {loss.item():.4f}')

    return total_loss / len(dataset)

if __name__ == "__main__":
    with open('./reverse_vocab.json', 'r') as f:
        reverse_vocab = json.load(f)

    model = TransformerCTC(
        input_dim=config.mfcc_feature,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        dropout=config.dropout
    ).to(config.device)

    criterion = nn.CTCLoss(blank=config.blank_token, zero_infinity=True,reduction='mean').to(config.device)
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
