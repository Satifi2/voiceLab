import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FinnalConfig import config
from FinalDataset import FinalDataset
import utils
import torch.nn.functional as F


class ConvTransformerCross(nn.Module):
    def __init__(self):
        super(ConvTransformerCross, self).__init__()
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
    
    def forward(self, encoder_input, decoder_input):
        encoder_input = self.encoder_pos(encoder_input)
        decoder_input = self.decoder_embedding(decoder_input)
        decoder_input = self.decoder_pos(decoder_input)

        transformer_output = self.transformer(
            src=encoder_input,
            tgt=decoder_input,
            src_key_padding_mask=encoder_input_pad,  
            tgt_key_padding_mask=decoder_input_pad, 
            memory_key_padding_mask=encoder_input_pad  
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
    
    optimizer = nn.optim.Adam(model.parameters(), lr=config.learning_rate__)
    
    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for npz_file in npz_files:
            dataset = FinalDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
            dataset_loss = 0
            for batch in dataloader:
                wav_filenames, encoder_input, decoder_input, decoder_expected_output = batch
                encoder_input = encoder_input.to(config.device)
                decoder_input = decoder_input.to(config.device)
                decoder_expected_output = decoder_expected_output.to(config.device)

                optimizer.zero_grad()
                # print(encoder_input[0].shape,decoder_input[0],decoder_expected_output[0])
                output = model(encoder_input, decoder_input)

                output = output.reshape(-1, output.shape[-1])
                decoder_expected_output = decoder_expected_output.reshape(-1)
                loss = config.criterion(output, decoder_expected_output)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                dataset_loss += loss.item()
            print(f"Epoch {epoch + 1},file:{npz_file},Total Loss: {total_loss / len(npz_files)}, dataset Loss {dataset_loss}")
            # print(wav_filenames[0],model.predict(encoder_input,decoder_input,reverse_vocab)[1][0])
            if dataset_loss < config.target_loss:
                utils.save_model_and_config(model, epoch, config.model_name__,model_save_dir)
        if (epoch+1) % 5 ==0:
            utils.save_model_and_config(model, epoch, config.model_name__,model_save_dir)
    