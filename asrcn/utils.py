import torch.nn as nn
import numpy as np
import torch 
import os
import config
import json

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe.to(config.device)

def test_position_encoding():
    d_model = 10
    max_len = 8
    pos_encoder = PositionalEncoding(max_len, d_model)

    batch_size = 2
    x = torch.zeros(batch_size,max_len, d_model)

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

def save_model_and_config(model, epoch, model_name, save_dir=os.path.join('..','model')):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model_save_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    config_save_path = os.path.join(save_dir, f'{model_name}_config.json')
    config_data = {
        'model_name': model_name,
        'model_name':config.model_name,
        'mfcc_feature': config.mfcc_feature,
        'max_sentence_len': config.max_sentence_len,
        'max_mfcc_seqlen': config.max_mfcc_seqlen,
        'num_attention_heads': config.num_attention_heads,
        'num_layers': config.num_layers,
        'ffn_hidden_dim': config.ffn_hidden_dim,
        'vocab_size': config.vocab_size,
        'device': config.device,
        'learning_rate': config.learning_rate,
        'dataloader_batch_size': config.dataloader_batch_size
    }
    with open(config_save_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"Configuration saved to {config_save_path}")

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    config.model_name = config_data['model_name']
    config.mfcc_feature = config_data['mfcc_feature']
    config.max_sentence_len = config_data['max_sentence_len']
    config.max_mfcc_seqlen = config_data['max_mfcc_seqlen']
    config.num_attention_heads = config_data['num_attention_heads']
    config.num_layers = config_data['num_layers']
    config.ffn_hidden_dim = config_data['ffn_hidden_dim']
    config.vocab_size = config_data['vocab_size']
    config.device = config_data['device']
    config.learning_rate = config_data['learning_rate']
    config.dataloader_batch_size = config_data['dataloader_batch_size']

    print("the configuration is loaded")

if __name__ == "__main__":
    test_position_encoding()
    test_pad_mask()

