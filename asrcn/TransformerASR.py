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
        
        transformer_output = self.transformer(encoder_input, decoder_input)
        
        output = self.fc_out(transformer_output)
        return output  # (batch_size, seq_len, vocab_size)

# 保存模型
def save_model(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

# 加载数据和训练模型
if __name__ == '__main__':
    train_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    save_path = os.path.join('..', 'save', 'model_checkpoint.pth')
    npz_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.npz')]
    
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
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for npz_file in npz_files:
            dataset = ASRDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for wav_filenames, encoder_input, decoder_input, decoder_expected_output in dataloader:
                encoder_input = encoder_input.to(config.device)
                decoder_input = decoder_input.to(config.device)
                decoder_expected_output = decoder_expected_output.to(config.device)
                
                optimizer.zero_grad()
                output = model(encoder_input, decoder_input)
                # 调整 output 形状为 (seq_len, batch_size, vocab_size)
                output = output.permute(1, 0, 2)
                
                # 计算CTC损失所需的长度信息
                input_lengths = torch.full((output.size(1),), output.size(0), dtype=torch.long).to(config.device)
                target_lengths = torch.sum(decoder_expected_output != 0, dim=1).to(config.device)
                
                # 将 target 展平为一维张量并移除填充元素
                target = decoder_expected_output[decoder_expected_output != 0].flatten()
                
                # 计算CTC损失
                loss = criterion(output, target, input_lengths, target_lengths)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            print(f"{npz_file} finished,Total Loss: {total_loss / len(npz_files)}")
    
    # 保存模型
    save_model(model, optimizer, num_epochs, total_loss / len(npz_files), save_path)
    print(f"Model saved to {save_path}")
