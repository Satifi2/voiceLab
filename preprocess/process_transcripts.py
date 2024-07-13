import os
import json


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
    max_length = 30
    print("pad bos eos",pad_id,bos_id,eos_id)

    processed_data = {}

    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ')
            key = parts[0]
            characters = ''.join(parts[1:])
            token_ids = [vocab[char] for char in characters]

            if len(token_ids) < max_length:
                token_ids.extend([pad_id] * (max_length - len(token_ids)))

            decoder_input = [bos_id] + token_ids
            decoder_expected_output = token_ids + [eos_id]

            processed_data[key] = {
                "decoder_input": ' '.join(map(str, decoder_input)),
                "decoder_expected_output": ' '.join(map(str, decoder_expected_output))
            }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"处理后的数据已保存到 {output_path}")

if __name__ == "__main__":
    process_transcripts()
