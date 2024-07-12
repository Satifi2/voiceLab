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
    vocab = {'pad': 0, 'bos': 1, 'eos': 2} # pad记作0的好处是，之后可能要对pad进行掩码，这样好区分
    current_index = 3
    max_tokens = 0
    total,cnt = 0,0

    # 词汇表不用打乱，影响小且后面都是无序的
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            total = total + 1
            parts = line.strip().split(' ')
            characters = ''.join(parts[1:])  # 中文语音识别不分词
            if len(characters) > 30 : #会发现绝大部分句子都很短
                print(len(characters),characters)
                cnt = cnt +1
            max_tokens = max(max_tokens, len(characters))
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

    print("文本序列最大长:", max_tokens)
    print("总句子数:", total, "长度大于30的句子数:", cnt,"占比:",cnt/total)
    print("词汇表大小:", len(vocab))
    print(f"词汇表已保存到 {vocab_path}")
    print(f"反转词汇表已保存到 {reverse_vocab_path}")

if __name__ == "__main__":
    preprocess_vocab()

'''
原来文本序列最大长: 44
总句子数: 141600 长度大于30的句子数: 21 占比: 0.0001483050847457627
'''