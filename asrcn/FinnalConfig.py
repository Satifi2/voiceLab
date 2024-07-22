import os
import json
import torch.nn as nn

def load_vocab(vocab_dir):
    with open(os.path.join(vocab_dir,'vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    with open(os.path.join(vocab_dir,'reverse_vocab.json'), 'r', encoding='utf-8') as f:
        reverse_vocab = json.load(f)
    vocab_list = [reverse_vocab[str(key)] for key in reverse_vocab.keys()]
    print(f"{__name__}: vocab loaded")
    return vocab, reverse_vocab, vocab_list


def load_transcript(file_path):
    transcript_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                key = parts[0]
                value = ''.join(parts[1:])
                transcript_dict[key] = value
    print(f"{__name__}: transcript loaded")
    return transcript_dict


class FinnalConfig:
    def __init__(self):
        self.seed = 42
        self.model_name__ = "transformer_conv_cross"
        self.mfcc_feature = 128
        self.max_sentence_len = 31
        self.max_mfcc_seqlen = 460
        self.num_attention_heads = 8
        self.model_dim = 512
        self.num_layers = 12
        self.ffn_hidden_dim = 12288
        self.vocab_size = 4336
        self.device = 'cuda'
        self.learning_rate__ = 1e-5
        self.weight_decay = 1e-6
        self.dataloader_batch_size = 64
        self.dropout = 0.1
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2 
        self.target_loss__ = 0.0
        self.__vocab__, self.__reverse_vocab__, self.__vocab_list__ = load_vocab(os.path.join('..','data','data_aishell','preprocessed'))
        self.__transcript__ = load_transcript('../data/data_aishell/transcript/aishell_transcript_v0.8.txt')
        self.__criterion__ = nn.CrossEntropyLoss(ignore_index=self.pad_token)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

config = FinnalConfig()

def test_load_vocab():
    print("vocab", config.__vocab__)
    print("reverse_vocab", config.__reverse_vocab__)
    print("vocab_list", config.__vocab_list__)


if __name__ == '__main__':
    print(config.to_dict())
    test_load_vocab()