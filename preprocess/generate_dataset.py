import json
import os

def load_processed_transcripts():
    input_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'processed_transcripts.json')

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

# 示例使用
if __name__ == "__main__":
    transcripts_dict = load_processed_transcripts()
    print(transcripts_dict['BAC009S0002W0122'])
    print("所有句子长度都是：",len(transcripts_dict['BAC009S0002W0122']['decoder_input'].split(' ')))
