from config import config
import utils
from transformer_asr import TransformerASR
from torch.utils.data import Dataset, DataLoader
import os
import torch
import json
from dataset import AishellDataset

if __name__ == "__main__":
    model_save_dir = os.path.join('..', 'model','crossentropy')

    model_save_path = os.path.join(model_save_dir,'transformer_crossentropy_epoch_4.pth')
    config_save_path = os.path.join(model_save_dir,"transformer_crossentropy_config.json")

    utils.load_config(config_save_path)
    print(f"the model {config.model_name} is loading")

    model = TransformerASR(
        input_dim=config.mfcc_feature,
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        dropout=config.dropout
    ).to(config.device)

    model.load_state_dict(torch.load(model_save_path, map_location=config.device))

    reverse_vocab_path = os.path.join('reverse_vocab.json')
    with open(reverse_vocab_path, 'r', encoding='utf-8') as f:
        reverse_vocab = json.load(f)

    model.eval()
    test_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'test')
    npz_files = [os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith('.npz')]

    base_dir = '../data/data_aishell/wav/train/'
    all_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]

    for epoch in range(1, 6):
        for directory in all_dirs:
            print(f"test on directory: {directory}")
            dataset = AishellDataset(directory)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            for i, (mfcc_matrix, char_ids, wav_file) in enumerate(dataloader):
                mfcc_matrix = mfcc_matrix.squeeze(0).to(config.device)
                char_ids = torch.tensor(char_ids).long().to(config.device)
                # print("mfcc_matrix",mfcc_matrix,mfcc_matrix.shape)
                # print("char_ids",char_ids,char_ids.shape)
        
                decoder_input = torch.cat([torch.tensor([config.bos_token], device=config.device), char_ids])
                target = torch.cat([char_ids, torch.tensor([config.eos_token], device=config.device)])
                print("decoder_input",decoder_input,decoder_input.shape)
                print("target",target,target.shape)
                output = model(mfcc_matrix,char_ids)
                prediction = output.argmax(-1)
                print("prediction",prediction,prediction.shape)
                if i==5:
                    break
            break
        break