from config import config
import utils
from TransformerASR import TransformerASR, model_init
from torch.utils.data import Dataset, DataLoader
import os
import torch
import json
from ASRDataset import ASRDataset

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

    for npz_file in npz_files:
        dataset = ASRDataset(npz_file)
        dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=False)
        for batch in dataloader:
            wav_filenames, encoder_input, decoder_input, _ = batch
            encoder_input = encoder_input.to(config.device)
            decoder_input = decoder_input.to(config.device)

            predicted_indices, predicted_words = model.predict(encoder_input, decoder_input, reverse_vocab)
            for i, filename in enumerate(wav_filenames):
                print(f"File: {filename}")
                print(f"Predicted Indices: {predicted_indices[i]}")
                print(f"Predicted Words: {' '.join(predicted_words[i])}")
            break
        break

