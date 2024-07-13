import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np
import librosa


# 定义数据集类
class AishellDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.keys = list(self.data.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.data[key]
        encoder_input = np.array(sample['encoder_input'])
        decoder_input = np.array(sample['decoder_input'])
        decoder_output = np.array(sample['decoder_expected_output'])

        return key, torch.tensor(encoder_input, dtype=torch.float32), torch.tensor(decoder_input,
                                                                                   dtype=torch.long), torch.tensor(
            decoder_output, dtype=torch.long)


# 定义Transformer模型
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
        # 添加位置编码
        src = src + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output


# 数据加载器
def collate_fn(batch):
    keys, encoder_inputs, decoder_inputs, decoder_outputs = zip(*batch)
    encoder_inputs = nn.utils.rnn.pad_sequence(encoder_inputs, batch_first=True)
    decoder_inputs = nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True)
    decoder_outputs = nn.utils.rnn.pad_sequence(decoder_outputs, batch_first=True)
    return keys, encoder_inputs, decoder_inputs, decoder_outputs


# 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for keys, encoder_inputs, decoder_inputs, decoder_outputs in dataloader:
            encoder_inputs = encoder_inputs.to(device)
            decoder_inputs = decoder_inputs.to(device)
            decoder_outputs = decoder_outputs.to(device)

            optimizer.zero_grad()
            outputs = model(encoder_inputs, decoder_inputs[:, :-1])
            outputs = outputs.permute(1, 0, 2)  # [T, N, C]
            loss = criterion(outputs, decoder_outputs[:, 1:])
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
    vocab_size = 4336  # 根据词汇表的大小调整
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    # 数据集和数据加载器
    base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    json_path = os.path.join(base_dir, 'S0002.json')
    dataset = AishellDataset(json_path)

    # 打印指定样本的decoder_input
    for key, encoder_input, decoder_input, decoder_output in dataset:
        if key == 'BAC009S0002W0122':
            print(f'Decoder input for {key}: {decoder_input}')
            break

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    # 模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(input_dim, embed_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward,
                             max_seq_length, vocab_size)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # 使用CTC损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, dataloader, criterion, optimizer, num_epochs, device)
