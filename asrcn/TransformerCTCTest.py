import os
import torch
import json
from config import config
import utils
from TransformerCTC import TransformerCTC
from ASRDataset import ASRDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    model_save_dir = os.path.join('..', 'model', 'transformer_ctc')
    model_save_path = os.path.join(model_save_dir, 'transformer_ctc_fix_epoch_0.pth')
    config_save_path = os.path.join(model_save_dir, 'transformer_ctc_fix_config.json')

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

    test_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    npz_files = [os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith('.npz')]

    for npz_file in npz_files:
        dataset = ASRDataset(npz_file)
        dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
        for batch in dataloader:
            wav_filenames, encoder_input, _, _ = batch
            encoder_input = encoder_input.to(config.device)

            predict_ids = model.predict(encoder_input)
            predict_words =  [[reverse_vocab[str(word_id)] for word_id in sentence] for sentence in predict_ids]
            print(predict_words)
            break
        break
