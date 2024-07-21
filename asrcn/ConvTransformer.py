import torch.nn as nn
import torch
import utils
from BetterConfig import config
from torch.utils.data import DataLoader
from BetterConfig import config
from BetterDataset import BetterDataset
from fast_ctc_decode import beam_search
import os
from torch.optim.lr_scheduler import CosineAnnealingLR

class ConvTransformerModel(nn.Module):
    def __init__(self):
        super(ConvTransformerModel, self).__init__()
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
        
        self.pos_encoder = utils.PositionalEncoding(max_len=config.max_mfcc_seqlen, d_model= config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
                        d_model=config.model_dim,
                        nhead=config.num_attention_heads,
                        dim_feedforward=config.ffn_hidden_dim, 
                        dropout=config.dropout, 
                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.fc = nn.Linear(config.model_dim, config.vocab_size)
        

    def get_conv_out_lens(self, input_lengths):
        seq_lens = input_lengths.clone()
        for m in self.conv.modules():
            if isinstance(m, (nn.Conv1d, nn.MaxPool1d)):
                padding = m.padding if isinstance(m.padding, int) else m.padding[0]
                dilation = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
                kernel_size = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                stride = m.stride if isinstance(m.stride, int) else m.stride[0]

                seq_lens = ((seq_lens + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        return seq_lens.int()

    def forward(self, input, input_lengths):
        input = input.transpose(1, 2)
        input = self.conv(input)
        input = input.transpose(1, 2)
        conv_out_lens = self.get_conv_out_lens(input_lengths)
        padding_mask = utils.create_padding_mask(conv_out_lens, input.size(1))
        input = self.pos_encoder(input)
        output = self.transformer_encoder(input, src_key_padding_mask=padding_mask)
        logits = self.fc(output)
        return logits, conv_out_lens
    
    def predict(self, output, output_lengths):
        # print("output in predict", output)
        # print("output_lengths in predict", output_lengths)
        with torch.no_grad():
            result = []
            posteriors = torch.nn.functional.softmax(output, dim=-1)
            posteriors_np = posteriors.cpu().numpy()
            batch_size = posteriors_np.shape[0]

            for i in range(batch_size):
                sample_posteriors = posteriors_np[i, :output_lengths[i], :]
                prediced_chars, path = beam_search(sample_posteriors, config.__vocab_list__, beam_size=config.beam_size, beam_cut_threshold=config.beam_cut_threshold)
                result.append(prediced_chars)

        return result
        
    
def train(model, criterion, optimizer, device, train_dir_path, save_dir, num_epochs=5):
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]
    npz_test_file = os.path.join(train_dir_path, 'S0002.npz')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for npz_file in npz_files:
            dataset = BetterDataset(npz_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=True)
            dataset_loss = 0
            
            for i, (wav_filenames, source, target, source_valid, target_valid) in enumerate(dataloader):
                source = source.to(device)
                target = target.to(device)
                source_lengths = torch.sum(source_valid, dim=1).to(device)
                target_lengths = torch.sum(target_valid, dim=1).to(device)

                optimizer.zero_grad()

                output, output_lengths = model(source, source_lengths)
                if i == 0:
                    # print("source.shape", source.shape)
                    # print("output.shape", output.shape)
                    # print("output[0]", output[0])
                    # print("output_lengths", output_lengths)
                    # print("target.shape",target.shape)
                    # print("target_lengths", target_lengths)
                    print(f"prediction: {model.predict(output[:3], output_lengths[:3])}")
                    print(f"groudtruth:{[config.__transcript__[filename] for filename in wav_filenames[:3]]}")
                output = nn.functional.log_softmax(output, dim=2)

                loss = criterion(output.transpose(0, 1), target, output_lengths, target_lengths)
                loss.backward()
                optimizer.step()
                dataset_loss += loss.item()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}, File {os.path.basename(npz_file)}, Average Loss: {dataset_loss}")
            # break
        print(f"Epoch {epoch+1} completed.")
        # break
        utils.save_model_and_config(model, epoch+1, config.model_name)

    print("Training completed.")

if __name__ == '__main__':

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = ConvTransformerModel().to(device)

    print("test saving")
    utils.save_model_and_config(model, 999, config.model_name)

    criterion = nn.CTCLoss(blank=config.blank_token, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    save_dir = os.path.join('..', 'model', 'conv_transformer')
    os.makedirs(save_dir, exist_ok=True)

    train(model, criterion, optimizer, device, train_dir_path, save_dir, num_epochs=5)