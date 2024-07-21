import os
import json


def load_vocab(vocab_dir):
    with open(os.path.join(vocab_dir,'vocab.json'), 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    with open(os.path.join(vocab_dir,'reverse_vocab.json'), 'r', encoding='utf-8') as f:
        reverse_vocab = json.load(f)
    vocab_list = [reverse_vocab[str(key)] for key in reverse_vocab.keys()]
    print("vocab loaded")
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
    print("transcript loaded")
    return transcript_dict


class BetterConfig:
    def __init__(self):
        self.seed = 42
        self.model_name = "transformer_ctc_conv"
        self.mfcc_feature = 128
        self.max_sentence_len = 30
        self.max_mfcc_seqlen = 460
        self.num_attention_heads = 8
        self.encoder_hidden_dim = 512
        self.model_dim = 512
        self.num_layers = 4
        self.ffn_hidden_dim = 4096
        self.vocab_size = 4336
        self.beam_size = 3
        self.beam_cut_threshold = 0
        self.device = 'cuda'
        self.learning_rate = 1e-6
        self.dataloader_batch_size = 64
        self.dropout = 0.1
        self.blank_token = 0
        self.bos_token = 1
        self.eos_token = 2 
        self.target_loss = 0.0
        self.__vocab__, self.__reverse_vocab__, self.__vocab_list__ = load_vocab(os.path.join('..','data','data_aishell','preprocessed'))
        self.__transcript__ = load_transcript('../data/data_aishell/transcript/aishell_transcript_v0.8.txt')
        self.__debugmode__ = -1

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

config = BetterConfig()

def test_load_vocab():
    print("vocab", config.__vocab__)
    print("reverse_vocab", config.__reverse_vocab__)
    print("vocab_list", config.__vocab_list__)


if __name__ == '__main__':
    print(config.to_dict())
    test_load_vocab()