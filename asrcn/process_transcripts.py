'''
文件说明:
首先读取vocab.json和reverse_vocab.json文件，从而知道每个词对应的编号
然后读取aishell_transcript_v0.8.txt文件，将句子转换为id序列
将id序列补齐为句子最大长度30，并且在前面加上bos和在后面加上eos，从而构建decoder_input和decoder_expected_output
最后将处理后的数据保存到processed_transcripts.json文件中。
'''

import os
import json
from config import config

def load_vocab():
    vocab_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'vocab.json')
    reverse_vocab_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'reverse_vocab.json')

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    with open(reverse_vocab_path, 'r', encoding='utf-8') as f:
        reverse_vocab = json.load(f)

    return vocab, reverse_vocab


def process_transcripts():
    transcript_path = os.path.join('..', 'data', 'data_aishell', 'transcript', 'aishell_transcript_v0.8.txt')
    output_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'processed_transcripts.json')

    vocab, _ = load_vocab()
    pad_id = vocab['pad']
    bos_id = vocab['bos']
    eos_id = vocab['eos']
    print("pad bos eos",pad_id,bos_id,eos_id)

    processed_data = {}

    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            key = parts[0]
            characters = ''.join(parts[1:])
            token_ids = [vocab[char] for char in characters]

            if len(token_ids) < config.max_sentence_len:
                token_ids.extend([pad_id] * (config.max_sentence_len - len(token_ids)))

            decoder_input = [bos_id] + token_ids
            decoder_expected_output = token_ids + [eos_id]

            processed_data[key] = {
                "decoder_input": decoder_input,
                "decoder_expected_output": decoder_expected_output
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"处理后的数据已保存到 {output_path}")
    return processed_data

if __name__ == "__main__":
    processed_data = process_transcripts()
    print(processed_data['BAC009S0002W0122'])

'''
pad bos eos 0 1 2
处理后的数据已保存到 ..\data\data_aishell\preprocessed\processed_transcripts.json
{'decoder_input': '1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0', 'decoder_expected_output': '3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2'}
'''