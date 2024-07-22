import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FinnalConfig import config
from FinalDataset import *
import utils
import torch.nn.functional as F


class ConvTransformerCross(nn.Module):
    def __init__(self):
        super(ConvTransformerCross, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=config.mfcc_feature, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2),  # Reduces length by 2
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, stride=1)
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
            activation=F.silu,
            batch_first=True
        ).to(config.device)
        self.fc_out = nn.Linear(config.model_dim, config.vocab_size).to(config.device)
        
    
    def forward(self, encoder_input, decoder_input, source_lengths, target_lengths):
        encoder_input = self.encoder_pos(encoder_input)
        decoder_input = self.decoder_embedding(decoder_input)
        decoder_input = self.decoder_pos(decoder_input)
        input = self.conv(encoder_input)
        input_lengths = self.get_conv_out_lens(source_lengths)
        src_padding_mask = create_padding_mask(input_lengths, input.shape[1])
        src_padding_mask = create_padding_mask(target_lengths, decoder_input.shape[1])

        transformer_output = self.transformer(
            src=encoder_input,
            tgt=decoder_input,
            src_key_padding_mask=src_padding_mask,  
            tgt_key_padding_mask=src_padding_mask, 
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
        for m in [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4]:
            if isinstance(m, (nn.Conv1d, nn.MaxPool1d)):
                padding = m.padding if isinstance(m.padding, int) else m.padding[0]
                dilation = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
                kernel_size = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                stride = m.stride if isinstance(m.stride, int) else m.stride[0]

                seq_lens = ((seq_lens + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        return seq_lens
    

    def predict(self, encoder_input, decoder_input, reverse_vocab):
        with torch.no_grad():
            output = self.forward(encoder_input, decoder_input)
            predicted_indices = torch.argmax(output, dim=-1)
            batch_size, seq_len = predicted_indices.size()
            predicted_words = []
            for i in range(batch_size):
                predicted_words.append([reverse_vocab[str(idx.item())] for idx in predicted_indices[i] if idx != config.pad_token])

            return predicted_indices, predicted_words


def model_init(model_save_path='', config_save_path=''):
    model = ConvTransformerCross()
    if model_save_path and config_save_path:
        utils.load_config(config_save_path)
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
        loss += config.criterion(valid_output, valid_target)
    return loss / batch_size 

def train(model, num_epochs = 5):
    optimizer = nn.optim.Adam(model.parameters(), lr=config.learning_rate__)
    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for npz_file in npz_files:
            dataset = FinalDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
            dataset_loss = 0
            for wav_filenames, source, decoder_input, target, source_invalid, target_invalid, source_lengths, target_lengths in dataloader:
                optimizer.zero_grad()
                output = model(source, decoder_input, source_lengths,target_lengths)
                loss = compute_loss(output, target, target_lengths)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                dataset_loss += loss.item()
            print(f"Epoch {epoch + 1},file:{npz_file},Total Loss: {total_loss / len(npz_files)}, dataset Loss {dataset_loss}")
            if dataset_loss < config.target_loss:
                utils.save_model_and_config(model, epoch, config.model_name__,model_save_dir)
        if (epoch+1) % 5 ==0:
            utils.save_model_and_config(model, epoch, config.model_name__,model_save_dir)

if __name__ == '__main__':
    utils.set_seed()
    model_save_dir = os.path.join('..', 'model','transformer_final')
    model_save_path = os.path.join(model_save_dir,'transformer_asr_51_epoch_0.pth')
    config_save_path = os.path.join(model_save_dir,"transformer_asr_51_config.json")
    model, reverse_vocab= model_init()

    #test save
    save_dir = os.path.join('..','temp')
    utils.save_model_and_config(model, 999, "test",save_dir)    
    print(f'{config.model_name__} is being trained with learning rate {config.learning_rate__}, the target loss is {config.target_loss__}')
    
    train(model=model)


    