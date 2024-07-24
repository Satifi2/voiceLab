import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FinnalConfig import config
from FinalDataset import *
import utils
import Utils
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR


class ConvTransformerCross(nn.Module):
    def __init__(self):
        super(ConvTransformerCross, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=config.mfcc_feature, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, stride=1)
        )

        self.encoder_pos = utils.PositionalEncoding(config.max_mfcc_seqlen, config.model_dim)
        self.decoder_embedding = nn.Embedding(config.vocab_size, config.model_dim, padding_idx=0).to(config.device)
        self.decoder_pos = utils.PositionalEncoding(config.max_sentence_len, config.model_dim)
        self.transformer = nn.Transformer(
            d_model=config.model_dim,
            nhead=config.num_attention_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.ffn_hidden_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        ).to(config.device)
        self.fc_out = nn.Linear(config.model_dim, config.vocab_size).to(config.device)
        
    
    def forward(self, source, decoder_input, source_lengths, target_lengths):
        source = source.transpose(1, 2)
        input = self.conv(source)
        input = input.transpose(1, 2)
        input_lengths = self.get_conv_out_lens(source_lengths)

        input = self.encoder_pos(input)
        decoder_input = self.decoder_embedding(decoder_input)
        decoder_input = self.decoder_pos(decoder_input)

        src_padding_mask = create_padding_mask(input_lengths, input.shape[1])
        tgt_padding_mask = create_padding_mask(target_lengths, decoder_input.shape[1])

        transformer_output = self.transformer(
            src=input,
            tgt=decoder_input,
            src_key_padding_mask=src_padding_mask,  
            tgt_key_padding_mask=tgt_padding_mask, 
            memory_key_padding_mask=src_padding_mask  
        )
        
        output = self.fc_out(transformer_output)
        return output
        
    def predict(self, encoder_input, decoder_input, reverse_vocab):
        batch_size = encoder_input.size(0)
        device = encoder_input.device
        decoder_input = torch.full((batch_size, 1), config.bos_token, dtype=torch.long, device=device)
        predicted_indices = []
        for _ in range(config.max_sentence_len):
            with torch.no_grad():
                output = self.forward(encoder_input, decoder_input)
            next_word = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
            # print(encoder_input.shape,decoder_input.shape,output.shape,next_word.shape)
            decoder_input = torch.cat([decoder_input, next_word], dim=-1)
            predicted_indices.append(next_word)
            # print(predicted_indices)

        predicted_indices = torch.cat(predicted_indices, dim=-1)
        # print(predicted_indices.shape)
        # print(predicted_indices)

        predicted_words = []
        for i in range(batch_size):
            words = []
            for idx in predicted_indices[i]:
                if idx != config.pad_token:
                    words.append(reverse_vocab[str(idx.item())])
            predicted_words.append(words)
        return predicted_indices, predicted_words
    

    def get_conv_out_lens(self, input_lengths):
        seq_lens = input_lengths.clone()
        for m in self.conv.modules():
            if isinstance(m, (nn.Conv1d, nn.MaxPool1d)):
                padding = m.padding if isinstance(m.padding, int) else m.padding[0]
                dilation = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
                kernel_size = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                stride = m.stride if isinstance(m.stride, int) else m.stride[0]

                seq_lens = ((seq_lens + 2 * padding - kernel_size) // stride) + 1
        return seq_lens.int()
    

    def predict(self, output):
        with torch.no_grad():
            predicted_indices = torch.argmax(output, dim=-1)
            batch_size, seq_len = predicted_indices.size()
            predicted_words = []
            for i in range(batch_size):
                predicted_words.append([config.__reverse_vocab__[str(idx.item())] for idx in predicted_indices[i] if idx != config.pad_token])

            print(predicted_indices[:2], predicted_words[:2])
            return predicted_indices, predicted_words


def model_init(model_save_path='', config_save_path=''):
    model = ConvTransformerCross().to(config.device)
    if model_save_path and config_save_path:
        Utils.load_config(config_save_path)
        model.load_state_dict(torch.load(model_save_path))
        print(f"The model's total parameter is {utils.model_parameters(model)}")
    return model


def compute_loss(output, target, target_lengths):
    loss = 0
    batch_size = output.size(0)
    for i in range(batch_size):
        valid_length = target_lengths[i]
        valid_output = output[i, :valid_length, :]
        valid_target = target[i, :valid_length]
        loss += config.__criterion__(valid_output, valid_target)
    return loss / batch_size 


def train(model, num_epochs = 20):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate__, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-15)
    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for npz_file in npz_files:
            dataset = FinalDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
            dataset_loss = 0
            for idx, (wav_filenames, source, decoder_input, target, source_invalid, target_invalid, source_lengths, target_lengths) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(source, decoder_input, source_lengths,target_lengths)
                loss = compute_loss(output, target, target_lengths)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                dataset_loss += loss.item()
                if idx == 0:
                    model.predict(output)
            print(f"Epoch {epoch + 1},file:{npz_file},Total Loss: {total_loss / len(npz_files)}, dataset Loss {dataset_loss}")
            if dataset_loss < config.target_loss__:
                Utils.save_model(model, config.model_name__, epoch, model_save_dir)

        Utils.save_model(model, config.model_name__, epoch, model_save_dir)
        scheduler.step()
        print(f"Epoch {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

if __name__ == '__main__':
    utils.set_seed()
    model_save_dir = os.path.join('..', 'model','transformer_final')
    model_save_path = os.path.join(model_save_dir,'transformer_equal_len_epoch_19.pth')
    config_save_path = os.path.join(model_save_dir,"transformer_equal_len_config.json")
    model= model_init(model_save_path, config_save_path)

    #test save
    save_dir = os.path.join('..','temp')
    Utils.save_model(model, 'temp', 999, save_dir)    
    print(f'{config.model_name__} is being trained with learning rate {config.learning_rate__}, the target loss is {config.target_loss__}')
    
    train(model=model)


    