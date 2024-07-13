'''
文件说明:
process_audio_files】用于处理一个文件夹及子文件夹内的所有文件，如果找到了对应的transcript【load_processed_transcripts】，就提取mfcc特征【extract_mfcc】
返回类似于 { "BAC009S0002W0122": { "encoder_input": [[], [], ...], "decoder_input": [], "decoder_expected_output": [] } } 的dataset
而save_dataset则将这个dataset保存到指定路径
最后我们调用process_all_folders，遍历data/data_aishell/wav/test下面的S0764,S0765,S0766...，分别生成datasset，保存到data/data_aishell/dataset/test下面
'''

import os
import json
import librosa

def load_processed_transcripts():
    input_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'processed_transcripts.json')

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def extract_mfcc(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    return mfcc.T  # 转置以得到 [时间, 特征] 格式

def process_audio_files(audio_dir, transcripts_dict):
    dataset = {}
    max_seq_length = 0
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                file_id = os.path.splitext(file)[0]
                if file_id in transcripts_dict:
                    wav_path = os.path.join(root, file)
                    mfcc_features = extract_mfcc(wav_path)
                    max_seq_length = max(max_seq_length, mfcc_features.shape[0])
                    decoder_input = list(map(int, transcripts_dict[file_id]["decoder_input"].split()))
                    decoder_expected_output = list(map(int, transcripts_dict[file_id]["decoder_expected_output"].split()))

                    dataset[file_id] = {
                        "encoder_input": mfcc_features.tolist(),
                        "decoder_input": decoder_input,
                        "decoder_expected_output": decoder_expected_output
                    }
    return dataset, max_seq_length

def save_dataset(dataset, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

def process_all_folders(base_audio_dir, transcripts_dict, output_base_dir):
    overall_max_seq_length = 0
    for folder in os.listdir(base_audio_dir):
        folder_path = os.path.join(base_audio_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder}")
            dataset, max_seq_length = process_audio_files(folder_path, transcripts_dict)
            overall_max_seq_length = max(overall_max_seq_length, max_seq_length)
            output_path = os.path.join(output_base_dir, f"{folder}.json")
            save_dataset(dataset, output_path)
            print(f"数据集已保存到 {output_path}","序列最大长度为:",max_seq_length)
    print(f"所有文件夹处理完成。整体最大序列长度为: {overall_max_seq_length}")

if __name__ == "__main__":
    transcripts_dict = load_processed_transcripts()
    print('BAC009S0002W0122', transcripts_dict['BAC009S0002W0122'])
    print("所有句子长度都是：", len(transcripts_dict['BAC009S0002W0122']['decoder_input'].split(' ')))

    # base_audio_dir = os.path.join('..', 'data', 'data_aishell', 'wav', 'train')
    # output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset','train')
    base_audio_dir = os.path.join('..', 'data', 'data_aishell', 'wav', 'dev')
    output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset','dev')
    # base_audio_dir = os.path.join('..', 'data', 'data_aishell', 'wav', 'test')
    # output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset','test')
    os.makedirs(output_base_dir, exist_ok=True)

    process_all_folders(base_audio_dir, transcripts_dict, output_base_dir)

'''
BAC009S0002W0122 {'decoder_input': '1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0', 'decoder_expected_output': '3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2'}
所有句子长度都是： 31
Processing folder: S0764
数据集已保存到 ..\data\data_aishell\dataset\test\S0764.json 序列最大长度为: 259
Processing folder: S0765
数据集已保存到 ..\data\data_aishell\dataset\test\S0765.json 序列最大长度为: 279
Processing folder: S0766
...
所有文件夹处理完成。整体最大序列长度为: 460
'''

