from FinnalConfig import config
import utils
from ConvTransformerCross import ConvTransformerCross, model_init
from torch.utils.data import Dataset, DataLoader
import os
from FinalDataset import *
from jiwer import cer

if __name__ == "__main__":
    utils.set_seed()
    model_save_dir = os.path.join('..', 'model','transformer_final')
    model_save_path = os.path.join(model_save_dir,'transformer_equal_len_d_epoch_4.pth')
    config_save_path = os.path.join(model_save_dir,"transformer_equal_len_d_config.json")
    model = model_init(model_save_path,config_save_path)
    model.eval()
    test_dir_path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'test')
    npz_files = [os.path.join(test_dir_path, f) for f in os.listdir(test_dir_path) if f.endswith('.npz')]
    transcript_path = '../data/data_aishell/transcript/aishell_transcript_v0.8.txt'
    transcript = utils.load_transcript(transcript_path)

    total_cer = 0.0
    total_iterations = 0

    for npz_file in npz_files:
        dataset = FinalDataset(npz_file)
        dataloader = DataLoader(dataset, batch_size=config.dataloader_batch_size, shuffle=False)
        for idx, (wav_filenames, source, decoder_input, target, source_invalid, target_invalid, source_lengths, target_lengths) in enumerate(dataloader):
            #teacher forcing
            # with torch.no_grad():
            #     output = model(source, decoder_input, source_lengths, target_lengths)
                # predicted_indices, predicted_words = model.predict(output)

            #auto regression
            # predicted_indices, predicted_words 
            groud_truth = transcript[wav_filenames[0]]
            print(f"groud truth of {wav_filenames[0]} containing {len(groud_truth)} words", groud_truth)
            print("the word id:",[config.__vocab__[char] for char in groud_truth])
            print("check target:",target[0])
            print("check target length", target_lengths[0])
            print("check decoder input", decoder_input[0])
            print("check input", source[0])
            print("check souce length", source_lengths[0])
            print("before predicting=====================================================")
            model.predict_auto_regression(source, source_lengths)
            break
        break
            # for i, filename in enumerate(wav_filenames):
            #     predicted_text = ''.join(predicted_words[i]).replace('eos', '')
            #     ground_truth_text = transcript[filename]
            #     cer_value = cer(ground_truth_text, predicted_text)

            #     if idx == 0 and i < 10:
            #         print(f"Predicted Words: {predicted_text}")
            #         print(f'Ground Truth: {ground_truth_text}')
            #         print(f'CER: {cer_value:.4f}')
            #     total_cer += cer_value
            #     total_iterations += 1

    # average_cer = total_cer / total_iterations
    # print(f'Average CER: {average_cer:.4f}')
    # print(f'Average CCR: {1-average_cer:.4f}')

