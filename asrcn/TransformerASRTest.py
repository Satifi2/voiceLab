from config import config
import utils
from TransformerASR import TransformerASR, model_init
from torch.utils.data import Dataset, DataLoader
import os
import torch
import json
from ASRDataset import ASRDataset
from jiwer import cer

if __name__ == "__main__":
    utils.set_seed()
    model_save_dir = os.path.join('..', 'model','transformer_asr')
    # model_save_path = os.path.join(model_save_dir,'transformer_asr_day_epoch_5.pth')
    # config_save_path = os.path.join(model_save_dir,"transformer_asr_day_config.json")

    # model_save_path = os.path.join(model_save_dir,'transformer_asr_night_epoch_4.pth')
    # config_save_path = os.path.join(model_save_dir,"transformer_asr_night_config.json")

    # model_save_path = os.path.join(model_save_dir,'transformer_asr_epoch_4.pth')
    # config_save_path = os.path.join(model_save_dir,"transformer_asr_config.json")

    # model_save_path = os.path.join(model_save_dir,'transformer_asr_norm_epoch_9.pth')
    # config_save_path = os.path.join(model_save_dir,"transformer_asr_norm_config.json")

    model_save_path = os.path.join(model_save_dir,'transformer_asr_51_epoch_0.pth')
    config_save_path = os.path.join(model_save_dir,"transformer_asr_51_config.json")

    model ,reverse_vocab= model_init(model_save_path,config_save_path)
    model.eval()
    test_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'test')
    npz_files = [os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith('.npz')]
    transcript_path = '../data/data_aishell/transcript/aishell_transcript_v0.8.txt'
    transcript = utils.load_transcript(transcript_path)

total_cer = 0.0
total_iterations = 0

for npz_file in npz_files:
    dataset = ASRDataset(npz_file)
    dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=False)
    for batch_idx, batch in enumerate(dataloader):
        wav_filenames, encoder_input, decoder_input, _ = batch
        encoder_input = encoder_input.to(config.device)
        decoder_input = decoder_input.to(config.device)

        predicted_indices, predicted_words = model.predict(encoder_input, decoder_input, reverse_vocab)
        for i, filename in enumerate(wav_filenames):
            predicted_text = ''.join(predicted_words[i]).replace('eos', '')
            ground_truth_text = transcript[filename]
            cer_value = cer(ground_truth_text, predicted_text)

            if batch_idx == 0 and i < 10:
                print(f"Predicted Words: {predicted_text}")
                print(f'Ground Truth: {ground_truth_text}')
                print(f'CER: {cer_value:.4f}')
            total_cer += cer_value
            total_iterations += 1

average_cer = total_cer / total_iterations
print(f'Average CER: {average_cer:.4f}')
print(f'Average CCR: {1-average_cer:.4f}')

