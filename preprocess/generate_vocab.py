'''
文件说明：
从data/data_aishell/transcript/aishell_transcript_v0.8.txt当中提取出所有的汉字
训练集和测试集当中的汉字都要在词汇表当中【共享】，防止出现不在词汇表当中的汉字，总的不到5000字，多构建一些也没关系
构建词汇表【从词汇到索引的映射】，存储到data/data_aishell/preprocessed/vocab.json
然后再存储一个反转词汇表【从索引到词汇的映射】存储到data/data_aishell/preprocessed/reverse_vocab.json
'''


import os
import json

def preprocess_vocab():
    # 路径定义,不用/\是因为这样可以在Linux和Windows上通用
    transcript_path = os.path.join('..', 'data', 'data_aishell', 'transcript', 'aishell_transcript_v0.8.txt')
    vocab_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'vocab.json')
    reverse_vocab_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'reverse_vocab.json')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    # 初始化词汇表，前三个词分别对应 pad, bos, eos，因为是中文，词汇中不会出现这三个词
    vocab = {'pad': 0, 'bos': 1, 'eos': 2}
    current_index = 3

    # 词汇表不用打乱，影响小且后面都是无序的
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) > 1:
                characters = ''.join(parts[1:])  # 中文语音识别不分词
                for char in characters:
                    if char not in vocab:
                        vocab[char] = current_index
                        current_index += 1

    reverse_vocab = {index: char for char, index in vocab.items()}

    # 将词汇表和反转词汇表保存到文件中，词汇表不占空间(几十KB)，所以不需要压缩,json很直观
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    with open(reverse_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(reverse_vocab, f, ensure_ascii=False, indent=4)

    print(f"词汇表已保存到 {vocab_path}")
    print(f"反转词汇表已保存到 {reverse_vocab_path}")

if __name__ == "__main__":
    preprocess_vocab()
