import os
import torch
import json
import config
import utils
from TransformerCTC import TransformerCTC
from ASRDataset import ASRDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    model_save_dir = os.path.join('..', 'model', 'transformer_ctc')
    model_save_path = os.path.join(model_save_dir, 'transformer_ctc_small_epoch_4.pth')
    config_save_path = os.path.join(model_save_dir, 'transformer_ctc_small_config.json')

    utils.load_config(config_save_path)
    print(f"The model {config.model_name} is loading")

    model = TransformerCTC(
        vocab_size=config.vocab_size,
        d_model=config.mfcc_feature,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        encoder_seqlen=config.max_mfcc_seqlen
    )

    model.load_state_dict(torch.load(model_save_path, map_location=config.device))
    model = model.to(config.device)
    model.eval()

    reverse_vocab_path = os.path.join('..', 'data', 'data_aishell', 'preprocessed', 'reverse_vocab.json')
    with open(reverse_vocab_path, 'r', encoding='utf-8') as f:
        reverse_vocab = json.load(f)

    test_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'test')
    npz_files = [os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith('.npz')]

    for npz_file in npz_files:
        dataset = ASRDataset(npz_file)
        dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=False)
        for batch in dataloader:
            wav_filenames, encoder_input, _, _ = batch
            encoder_input = encoder_input.to(config.device)

            predictions = model.predict(encoder_input)
            print(encoder_input.shape,predictions[:5])
            for i, filename in enumerate(wav_filenames):
                predicted_indices = predictions[i]
                predicted_words = [reverse_vocab[str(idx)] for idx in predicted_indices]
                print(f"File: {filename}")
                print(f"Predicted Indices: {predicted_indices}")
                print(f"Predicted Words: {' '.join(predicted_words)}")
            break
        break
