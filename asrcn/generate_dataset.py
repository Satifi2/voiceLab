'''
文件说明:
每一个S0764,S0765...文件夹生成一个数据集
('..', 'data', 'data_aishell', 'wav', 'test')当中包含很多文件夹，例如S0764,S0765...，其中每一个文件夹当中有很多文件，例如BAC009S0764W0121.wav
我们的目标是对于S0764,S0765...,我们都在('..', 'data', 'data_aishell', 'dataset','test')下面生成一个同名文件S0764.npz,S0765.npz
里面存储三个张量encoder_input，decoder_input，decoder_expected_output，张量当中包含很多样本
'''

import os
import json
import librosa
import numpy as np
from config import config
import torchaudio


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


def extract_fbank(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type='hanning',
        num_mel_bins=config.__fbank_feature__,
        dither=0.0,
        frame_shift=10
    )
    fbank = fbank.numpy()  
    return fbank


def extract_melspectrogram(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=config.__melspec_feature__
    )(waveform)
    return mel_spectrogram


def process_audio_files(audio_dir, transcripts_dict):
    encoder_inputs = []
    decoder_inputs = []
    decoder_expected_outputs = []
    wav_filenames = []

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_file = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                
                if file_id not in transcripts_dict:
                    continue
                wav_filenames.append(file_id)
                
                mfcc = extract_mfcc(wav_file)
                encoder_inputs.append(mfcc)
                
                decoder_input = transcripts_dict[file_id]['decoder_input']
                decoder_expected_output = transcripts_dict[file_id]['decoder_expected_output']
                
                decoder_inputs.append(decoder_input)
                decoder_expected_outputs.append(decoder_expected_output)
    
    encoder_input_array = np.array(encoder_inputs)
    decoder_input_array = np.array(decoder_inputs)
    decoder_expected_output_array = np.array(decoder_expected_outputs)

    return wav_filenames, encoder_input_array, decoder_input_array, decoder_expected_output_array


def process_all_folders(base_audio_dir, transcripts_dict, output_base_dir):
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        
    for folder_name in os.listdir(base_audio_dir):
        folder_path = os.path.join(base_audio_dir, folder_name)
        if os.path.isdir(folder_path):
            wav_filenames, encoder_input_array, decoder_input_array, decoder_expected_output_array = process_audio_files(folder_path, transcripts_dict)
            
            output_file_path = os.path.join(output_base_dir, f'{folder_name}.npz')
            np.savez_compressed(output_file_path,
                                wav_filenames=wav_filenames,
                                encoder_input=encoder_input_array,
                                decoder_input=decoder_input_array,
                                decoder_expected_output=decoder_expected_output_array)

            print("encoder_input=",encoder_input_array.shape,"\ndecoder_input=",decoder_input_array.shape,"\ndecoder_expected_output=",decoder_expected_output_array.shape)
            print(f'{folder_path}中的wav文件特征被提取出,并被存储到了{output_file_path}')


def test_extract_feature(wav_path):
    print(extract_fbank(wav_path).shape)
    print(extract_melspectrogram(wav_path).shape)
    print(extract_mfcc(wav_path).shape)


if __name__ == "__main__":
    transcripts_dict = load_processed_transcripts()
    print('BAC009S0002W0122', transcripts_dict['BAC009S0002W0122'])
    print("所有句子长度都是：", len(transcripts_dict['BAC009S0002W0122']['decoder_input']))

    # base_audio_dir = os.path.join('..', 'data', 'data_aishell', 'wav', 'train')
    # output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')

    # base_audio_dir = os.path.join('..', 'data', 'data_aishell', 'wav', 'dev')
    # output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'dev')

    base_audio_dir = os.path.join('..', 'data', 'data_aishell', 'wav', 'test')
    output_base_dir = os.path.join('..', 'data', 'data_aishell', 'dataset', 'test')

    test_extract_feature(os.path.join(base_audio_dir,'S0764','BAC009S0764W0121.wav'))
    # process_audio_files(os.path.join(base_audio_dir,'S0002'), transcripts_dict)
    process_all_folders(base_audio_dir,transcripts_dict,output_base_dir)

'''
(base) ubuntu@VM-0-15-ubuntu:~/voiceLab/asrcn$ python /home/ubuntu/voiceLab/asrcn/generate_dataset.py
BAC009S0002W0122 {'decoder_input': [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'decoder_expected_output': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]}
所有句子长度都是： 31
encoder_input= (360, 460, 128) 
decoder_input= (360, 31) 
decoder_expected_output= (360, 31)
../data/data_aishell/wav/train/S0247中的wav文件特征被提取出,并被存储到了../data/data_aishell/dataset/train/S0247.npz
encoder_input= (357, 460, 128) 
decoder_input= (357, 31) 
decoder_expected_output= (357, 31)
'''

'''
# process_audio_files(os.path.join(base_audio_dir,'S0002'), transcripts_dict)#打印其中的文件:
BAC009S0002W0137
BAC009S0002W0235
BAC009S0002W0242
BAC009S0002W0480
...
'''