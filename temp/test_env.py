import torch
import librosa
import jiwer

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
