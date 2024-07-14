import os
import json
import librosa
import numpy as np
import config

def load_processed_transcripts():
    input_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'processed_transcripts.json')
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_mfcc(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.mfcc_feature).T
    if mfcc.shape[0] < config.max_mfcc_seqlen:
        padding = np.zeros((config.max_mfcc_seqlen - mfcc.shape[0], config.mfcc_feature))
        mfcc = np.vstack((mfcc, padding))
    return mfcc

def process_audio_files(audio_dir, transcripts_dict):
    encoder_inputs = []
    decoder_inputs = []
    decoder_expected_outputs = []

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                
                if file_id not in transcripts_dict:
                    continue
                
                mfcc = extract_mfcc(wav_file)
                encoder_inputs.append(mfcc)
                
                decoder_input = transcripts_dict[file_id]['decoder_input']
                decoder_expected_output = transcripts_dict[file_id]['decoder_expected_output']
                
                decoder_inputs.append(decoder_input)
                decoder_expected_outputs.append(decoder_expected_output)
    
    encoder_input_array = np.array(encoder_inputs)
    decoder_input_array = np.array(decoder_inputs)
    decoder_expected_output_array = np.array(decoder_expected_outputs)

    return encoder_input_array, decoder_input_array, decoder_expected_output_array

def process_all_folders(base_audio_dir, transcripts_dict, output_base_dir):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        
    for folder_name in os.listdir(base_audio_dir):
        folder_path = os.path.join(base_audio_dir, folder_name)
        if os.path.isdir(folder_path):
            encoder_input_array, decoder_input_array, decoder_expected_output_array = process_audio_files(folder_path, transcripts_dict)
            
            output_file_path = os.path.join(output_base_dir, f'{folder_name}.npz')
            np.savez_compressed(output_file_path,
                                encoder_input=encoder_input_array,
                                decoder_input=decoder_input_array,
                                decoder_expected_output=decoder_expected_output_array)
            print("encoder_input=",encoder_input_array.shape,"\ndecoder_input=",decoder_input_array.shape,"\ndecoder_expected_output=",decoder_expected_output_array.shape)
            print(f'{folder_path}中的wav文件特征被提取出,并被存储到了{output_file_path}')

if __name__ == "__main__":
    transcripts_dict = load_processed_transcripts()
    print('BAC009S0764W0121', transcripts_dict['BAC009S0764W0121'])
    print("所有句子长度都是：", len(transcripts_dict['BAC009S0764W0121']['decoder_input']))

    base_audio_dir = os.path.join('..', 'data', 'data_aishell', 'wav', 'train')
    output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')

    process_all_folders(base_audio_dir, transcripts_dict, output_base_dir)
