import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from BetterConfig import config
from BetterDataset import BetterDataset
from fast_ctc_decode import beam_search
import utils

class ConvRNN(nn.Module):
    def __init__(self, input_features, rnn_hidden_size, rnn_type='lstm'):
        super(ConvRNN, self).__init__()
        encoder_hidden_dim = config.encoder_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=encoder_hidden_dim, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(encoder_hidden_dim),
            # nn.ReLU(),
            nn.Conv1d(in_channels=encoder_hidden_dim, out_channels=encoder_hidden_dim, kernel_size=8, stride=6, padding=1),
            # nn.BatchNorm1d(encoder_hidden_dim),
            # nn.ReLU()
        )

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=encoder_hidden_dim, hidden_size=rnn_hidden_size, batch_first=True, num_layers=config.num_layers)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=encoder_hidden_dim, hidden_size=rnn_hidden_size, batch_first=True, num_layers=config.num_layers)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=encoder_hidden_dim, hidden_size=rnn_hidden_size, batch_first=True, num_layers=config.num_layers)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        self.fc = nn.Linear(rnn_hidden_size, config.vocab_size)

    def get_conv_out_lens(self, input_lengths):
        seq_lens = input_lengths.clone()
        for m in self.conv.modules():
            if isinstance(m, nn.Conv1d):
                seq_lens = ((seq_lens + 2 * m.padding[0] - m.dilation[0] * (m.kernel_size[0] - 1) - 1) // m.stride[0]) + 1
        return seq_lens.int()

    def forward(self, input, input_lengths):
        # print("input.shape before", input.shape)
        input = input.transpose(1, 2)
        input = self.conv(input)

        input = input.transpose(1, 2)

        conv_out_lens = self.get_conv_out_lens(input_lengths)

        # print("input.shape after", input.shape)
        # print("conv_out_lens", conv_out_lens)

        packed_input = pack_padded_sequence(input, conv_out_lens, batch_first=True, enforce_sorted=False)

        if isinstance(self.rnn, nn.TransformerEncoder):
            output = self.rnn(input)
        else:
            packed_output, _ = self.rnn(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            # print("pad_packed_sequence output", output[0])

        output = self.fc(output)

        return output, conv_out_lens
    
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


def train(model, criterion, optimizer, device, train_dir_path, save_dir, config, num_epochs=5):
    npz_files = [os.path.join(train_dir_path, f) for f in os.listdir(train_dir_path) if f.endswith('.npz')]
    npz_test_file = os.path.join(train_dir_path, 'S0002.npz')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for npz_file in npz_files:
            dataset = BetterDataset(npz_test_file)
            dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=False)#temporal for mistake checking
            dataset_loss = 0
            
            for wav_filenames, source, target, source_valid, target_valid in dataloader:
                source = source.to(device)
                target = target.to(device)
                source_lengths = torch.sum(source_valid, dim=1)
                target_lengths = torch.sum(target_valid, dim=1)

                optimizer.zero_grad()

                output, output_lengths = model(source, source_lengths)
                print("source.shape", source.shape)
                print("output.shape", output.shape)
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
            print()
            # break
        print(f"Epoch {epoch+1} completed.")
        # break

    print("Training completed.")

if __name__ == '__main__':
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = ConvRNN(input_features=config.mfcc_feature, 
                         rnn_hidden_size=config.encoder_hidden_dim, 
                         rnn_type='lstm').to(device)

    criterion = nn.CTCLoss(blank=config.blank_token, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    save_dir = os.path.join('..', 'model', 'conv_transformer')
    os.makedirs(save_dir, exist_ok=True)

    train(model, criterion, optimizer, device, train_dir_path, save_dir, config, num_epochs=5)