import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from config import config
from ASRDataset import *
import utils

class TransformerASR(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, encoder_seqlen, decoder_seqlen):
        super(TransformerASR, self).__init__()
        self.encoder_pos = utils.PositionalEncoding(encoder_seqlen, d_model)
        self.encoder_norm = nn.LayerNorm(d_model).to(config.device)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0).to(config.device)
        self.decoder_pos = utils.PositionalEncoding(decoder_seqlen, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        ).to(config.device)
        self.fc_out = nn.Linear(d_model, vocab_size).to(config.device)
    
    def forward(self, encoder_input, decoder_input):
        # encoder_input_pad = utils.pad_mask(encoder_input)
        # decoder_input_pad = utils.pad_mask(decoder_input)

        # encoder_input = self.encoder_norm(encoder_input)
        encoder_input = self.encoder_pos(encoder_input)
        
        decoder_input = self.decoder_embedding(decoder_input)
        decoder_input = self.decoder_pos(decoder_input)
        
        transformer_output = self.transformer(
            src=encoder_input,
            tgt=decoder_input,
            # src_key_padding_mask=encoder_input_pad,  
            # tgt_key_padding_mask=decoder_input_pad, 
            # memory_key_padding_mask=encoder_input_pad  
        )
        
        output = self.fc_out(transformer_output)
        return output

    def predict(self, encoder_input, decoder_input, reverse_vocab):
        with torch.no_grad():
            output = self.forward(encoder_input, decoder_input)
            predicted_indices = torch.argmax(output, dim=-1)
            batch_size, seq_len = predicted_indices.size()
            predicted_words = []
            for i in range(batch_size):
                predicted_words.append([reverse_vocab[str(idx.item())] for idx in predicted_indices[i]])

            return predicted_indices, predicted_words
        
    def predict_(self, encoder_input, reverse_vocab, max_length=config.max_sentence_len):
        batch_size = encoder_input.size(0)
        device = encoder_input.device
        decoder_input = torch.full((batch_size, 1), config.bos_token, dtype=torch.long, device=device)
        predicted_indices = []
        for _ in range(max_length):
            with torch.no_grad():
                output = self.forward(encoder_input, decoder_input)

            next_word = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            decoder_input = torch.cat([decoder_input, next_word], dim=-1)
            predicted_indices.append(next_word)
            if (next_word == config.eos_token).all():
                break

        predicted_indices = torch.cat(predicted_indices, dim=-1)

        predicted_words = []
        for i in range(batch_size):
            words = []
            for idx in predicted_indices[i]:
                if idx.item() == config.eos_token:
                    break
                words.append(reverse_vocab[str(idx.item())])
            predicted_words.append(words)
        return predicted_indices, predicted_words


def model_init(model_save_path='', config_save_path=''):
    model = TransformerASR(
        vocab_size=config.vocab_size,
        d_model=config.mfcc_feature,
        nhead=config.num_attention_heads,
        num_encoder_layers=config.num_layers,
        num_decoder_layers=config.num_layers,
        dim_feedforward=config.ffn_hidden_dim,
        encoder_seqlen=config.max_mfcc_seqlen,
        decoder_seqlen=config.max_sentence_len
    )
    learning_rate, model_name, target_loss = config.learning_rate, config.model_name, config.target_loss
    if model_save_path and config_save_path:
        utils.load_config(config_save_path)
        model.load_state_dict(torch.load(model_save_path, map_location=config.device))
        print(f"The model {config.model_name} is loading")
        config.learning_rate, config.model_name, config.target_loss= learning_rate, model_name, target_loss
    model = model.to(config.device)
    return model


if __name__ == '__main__':
    utils.set_seed()
    model_save_dir = os.path.join('..', 'model','transformer_asr')
    model_save_path = os.path.join(model_save_dir,'transformer_asr_51t_epoch_4.pth')
    config_save_path = os.path.join(model_save_dir,"transformer_asr_51t_config.json")
    model = model_init(model_save_path,config_save_path)

    #test save
    save_dir = os.path.join('..','temp')
    utils.save_model_and_config(model, 999, "test",save_dir)    
    print(f'{config.model_name} is being trained with learning rate {config.learning_rate}, the target loss is {config.target_loss}')
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]
    save_dir = os.path.join('..','model','transformer_asr')

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for npz_file in npz_files:
            dataset = ASRDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
            dataset_loss = 0
            for batch in dataloader:
                wav_filenames, encoder_input, decoder_input, decoder_expected_output = batch
                encoder_input = encoder_input.to(config.device)
                decoder_input = decoder_input.to(config.device)
                decoder_expected_output = decoder_expected_output.to(config.device)

                optimizer.zero_grad()
                output = model(encoder_input, decoder_input)

                output = output.reshape(-1, output.shape[-1])
                decoder_expected_output = decoder_expected_output.reshape(-1)
                loss = criterion(output, decoder_expected_output)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                dataset_loss += loss.item()
            print(f"Epoch {epoch + 1},file:{npz_file},Total Loss: {total_loss / len(npz_files)}, dataset Loss {dataset_loss}")
            if dataset_loss < config.target_loss:
                utils.save_model_and_config(model, epoch, config.model_name,save_dir)
        if (epoch+1) % 5 ==0:
            utils.save_model_and_config(model, epoch, config.model_name,save_dir)
    