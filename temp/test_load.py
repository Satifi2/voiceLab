import numpy as np
import os
path = os.path.join('..', 'data', 'data_aishell', 'dataset', 'train', 'S0002.npz')
data = np.load(path)
print(data['wav_filenames'])
print(data['encoder_input'])
print(data['decoder_input'])
print(data['decoder_expected_output'])