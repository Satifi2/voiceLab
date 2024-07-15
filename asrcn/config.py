'''代码当中的参数值在这里定义，修改容易且避免不一致'''
import torch
mfcc_feature = 128
max_sentence_len = 31
max_mfcc_seqlen = 460
num_attention_heads = 8
num_layers = 6
ffn_hidden_dim = 1024
vocab_size = 4336
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
