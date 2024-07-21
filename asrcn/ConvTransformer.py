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
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=config.mfcc_feature, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=1, stride=1)
        
        # Pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Activation function
        self.silu = nn.SiLU()
        
        # Dropout
        self.dropout = nn.Dropout(p=0.1)  # You can adjust the dropout rate
        
        # Rest of the model
        self.pos_encoder = utils.PositionalEncoding(max_len=config.max_mfcc_seqlen, d_model=config.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
                        d_model=config.model_dim,
                        nhead=config.num_attention_heads,
                        dim_feedforward=config.ffn_hidden_dim, 
                        dropout=config.dropout, 
                        batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.fc = nn.Linear(config.model_dim, config.vocab_size)

    def get_conv_out_lens(self, input_lengths):
        # This method remains the same
        seq_lens = input_lengths.clone()
        for m in [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4]:
            if isinstance(m, (nn.Conv1d, nn.MaxPool1d)):
                padding = m.padding if isinstance(m.padding, int) else m.padding[0]
                dilation = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
                kernel_size = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                stride = m.stride if isinstance(m.stride, int) else m.stride[0]

                seq_lens = ((seq_lens + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        return seq_lens.int()

    def forward(self, input, input_lengths):
        # Convolutional layers with residual connections and dropout
        input = input.transpose(1, 2)
        
        # First conv block
        x = self.conv1(input)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.pool1(x)
        # residual = x
        
        # Second conv block
        x = self.conv2(x)
        x = self.silu(x)
        x = self.dropout(x)
        x = self.pool2(x)
        # x = x + nn.functional.interpolate(residual, size=x.shape[2], mode='linear', align_corners=False)
        # residual = x
        
        # Third conv block
        x = self.conv3(x)
        x = self.silu(x)
        x = self.dropout(x)
        # x = x + residual
        # residual = x
        
        # Fourth conv block
        x = self.conv4(x)
        x = self.silu(x)
        x = self.dropout(x)
        # x = x + nn.functional.interpolate(residual, size=x.shape[2], mode='linear', align_corners=False)
        
        # Transpose back
        x = x.transpose(1, 2)
        
        # Get conv output lengths
        conv_out_lens = self.get_conv_out_lens(input_lengths)
        
        # Create padding mask
        padding_mask = utils.create_padding_mask(conv_out_lens, x.size(1))
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Final linear layer
        logits = self.fc(output)
        
        return logits, conv_out_lens
        

    def get_conv_out_lens(self, input_lengths):
        # This method remains the same
        seq_lens = input_lengths.clone()
        for m in [self.conv1, self.pool1, self.conv2, self.pool2, self.conv3, self.conv4]:
            if isinstance(m, (nn.Conv1d, nn.MaxPool1d)):
                padding = m.padding if isinstance(m.padding, int) else m.padding[0]
                dilation = m.dilation if isinstance(m.dilation, int) else m.dilation[0]
                kernel_size = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                stride = m.stride if isinstance(m.stride, int) else m.stride[0]

                seq_lens = ((seq_lens + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        return seq_lens.int()


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
        utils.save_model_and_config(model, epoch+1, config.model_name, save_dir=os.path.join('..', 'model','conv_transformer'))

    print("Training completed.")

if __name__ == '__main__':

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    model = ConvTransformerModel().to(device)

    print("test saving")
    utils.save_model_and_config(model, 999, config.model_name, save_dir=os.path.join('..', 'model','conv_transformer'))

    criterion = nn.CTCLoss(blank=config.blank_token, reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

    train_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train')
    save_dir = os.path.join('..', 'model', 'conv_transformer')
    os.makedirs(save_dir, exist_ok=True)

    train(model, criterion, optimizer, device, train_dir_path, save_dir, num_epochs=5)