import torch.nn as nn
import numpy as np
import torch 
import os
from BetterConfig import config
import json
import random
import inspect

def set_seed():
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(config.device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print("x.shape",x.shape) #torch.Size([64, 460, 128])
        # print("pe.shape",self.pe.shape) #torch.Size([1, 460, 128])
        return x + self.pe[:,:x.shape[1],:]


def test_position_encoding():
    d_model = 10
    max_len = 8
    pos_encoder = PositionalEncoding(max_len, d_model)

    batch_size = 2
    x = torch.zeros(batch_size,max_len, d_model).to(config.device)

    x_encoded = pos_encoder(x)

    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    frequence = position*div_term
    print("position",position,position.shape)
    print("div_term",div_term,div_term.shape)
    print("frequence",frequence,frequence.shape)

    print("Input Tensor:",x,x.shape)
    print("Positional Encoded Tensor:",x_encoded,x_encoded.shape)

def pad_mask(input_tensor):
    if input_tensor.ndim == 3:  
        return torch.all(input_tensor == 0, dim=-1)
    elif input_tensor.ndim == 2:  
        return input_tensor == 0
    else:
        raise ValueError("Unsupported input tensor dimensions")

def test_pad_mask():
    input_tensor_3d = torch.tensor([
        [[1, 2, 3], [3, 5, 0], [0, 0, 0]],
        [[7, 8, 4], [0, 0, 0], [0, 0, 0]]
    ])
    print("3D Input Pad Mask:\n", pad_mask(input_tensor_3d), '\n\n')

    input_tensor_2d = torch.tensor([
        [1, 2, 3],
        [3, 4, 0],
        [1, 0, 0]
    ])
    print("2D Input Pad Mask:\n", pad_mask(input_tensor_2d))


def save_model_and_config(model, epoch, model_name, save_dir=os.path.join('..','model','transformer_asr')):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model_save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    config_save_path = os.path.join(save_dir, f'{model_name}_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=4)
    print(f"Configuration saved to {config_save_path}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    for key, value in config_data.items():
        setattr(config, key, value)
    
    print("The configuration is loaded")


def printf(parameter, comment='',detailed_level =1):
    if detailed_level == config.__debugmode__ :
        print(comment, parameter)


def load_transcript(file_path):
    transcript_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                key = parts[0]
                value = ''.join(parts[1:])
                transcript_dict[key] = value
    print("transcript loaded")
    return transcript_dict


def model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    return total_params


def create_padding_mask(lengths, max_len):
    return torch.arange(max_len).to(config.device).expand(len(lengths), max_len) >= lengths.unsqueeze(1)


def test_create_pad():
    lengths = torch.tensor([3,2,1]).to(config.device)
    print(create_padding_mask(lengths=lengths,max_len=5))

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":
    test_position_encoding()
    test_pad_mask()
    test_create_pad()

