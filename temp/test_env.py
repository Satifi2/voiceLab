import torch
import librosa
import jiwer
import numpy as np
from fast_ctc_decode import beam_search, viterbi_search

# 打印 PyTorch 版本
print("PyTorch version:", torch.__version__)

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

# 打印 CUDA 版本
if cuda_available:
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name:", torch.cuda.get_device_name(i))

print(jiwer.cer('123415','123425'))

alphabet = ["N","A","C","G","T"]
posteriors = np.random.rand(100, len(alphabet)).astype(np.float32)
seq, path = viterbi_search(posteriors, alphabet)

seq, path = beam_search(posteriors, alphabet, beam_size=1, beam_cut_threshold=0.1)
print(seq,len(seq))
print(path, len(path))
class_indices = np.argmax(posteriors[path], axis=1)
print(class_indices, len(class_indices))
print(''.join([alphabet[class_idx] for class_idx in class_indices]))