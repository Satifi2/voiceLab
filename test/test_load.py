import numpy as np
import os
path = os.path.join('..', 'data', 'data_aishell', 'names', 'train', 'S0002.npz')
name = np.load(path)
print(name['wav_filenames'])