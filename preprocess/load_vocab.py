import os
import json

#使用json的另一优点是读取一行代码搞定，txt还要设计如何解析
def load_vocab():
    vocab_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'vocab.json')
    reverse_vocab_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'reverse_vocab.json')

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    with open(reverse_vocab_path, 'r', encoding='utf-8') as f:
        reverse_vocab = json.load(f)

    return vocab, reverse_vocab

if __name__ == "__main__":
    vocab, reverse_vocab = load_vocab()

    print("词汇表:", vocab)
    print("反转词汇表:", reverse_vocab)
