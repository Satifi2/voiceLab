import torch.nn as nn
import numpy as np
import torch 

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
        return x + self.pe

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

if __name__ == "__main__":
    test_position_encoding()
    test_pad_mask()

