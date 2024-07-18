import os
import json

def preprocess_vocab():
    # 路径定义,不用/\是因为这样可以在Linux和Windows上通用
    transcript_path = os.path.join('..', 'data', 'data_aishell', 'transcript', 'aishell_transcript_v0.8.txt')
    vocab_path = os.path.join('.', 'vocab.json')
    reverse_vocab_path = os.path.join('.', 'reverse_vocab.json')
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)

    # 初始化词汇表，前三个词分别对应 pad, bos, eos，因为是中文，词汇中不会出现这三个词
    vocab = {'blank': 0, 'bos': 1, 'eos': 2} # pad记作0的好处是，之后可能要对pad进行掩码，这样好区分
    current_index = 3
    max_tokens = 0
    total_sentences = 0

    # 词汇表不用打乱，影响小且后面都是无序的
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_sentences += 1
            parts = line.strip().split(' ')
            characters = ''.join(parts[1:])  # 中文语音识别不分词
            max_tokens = max(max_tokens, len(characters))
            for char in characters:
                if char not in vocab:
                    vocab[char] = current_index
                    current_index += 1

    reverse_vocab = {index: char for char, index in vocab.items()}

    # 将词汇表和反转词汇表保存到文件中，词汇表不占空间(几十KB)，所以不需要压缩,json很直观且好读取
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    with open(reverse_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(reverse_vocab, f, ensure_ascii=False, indent=4)

    print("句子总数:", total_sentences)
    print("句子最大长:", max_tokens)
    print("词汇表大小:", len(vocab))
    print(f"词汇表已保存到 {vocab_path}")
    print(f"反转词汇表已保存到 {reverse_vocab_path}")

if __name__ == "__main__":
    preprocess_vocab()
