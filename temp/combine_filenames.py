import os
import numpy as np

# 定义文件夹路径
base_path = '/home/ubuntu/voiceLab/data/data_aishell'
names_path = os.path.join(base_path, 'names/train')
data_path = os.path.join(base_path, 'dataset/train')
output_path = os.path.join(base_path, 'dataset/train')

# 获取文件名列表
name_files = sorted([f for f in os.listdir(names_path) if f.endswith('.npz')])
data_files = sorted([f for f in os.listdir(data_path) if f.endswith('.npz')])

# 确保每个文件都有对应的匹配
if len(name_files) != len(data_files):
    raise ValueError("The number of files in the two directories do not match.")

# 合并文件
for name_file, data_file in zip(name_files, data_files):
    if name_file != data_file:
        raise ValueError(f"File mismatch: {name_file} and {data_file} do not match.")
    
    # 加载names文件
    names_data = np.load(os.path.join(names_path, name_file))
    wav_filenames = names_data['wav_filenames']

    # 加载data文件
    data = np.load(os.path.join(data_path, data_file))
    encoder_input = data['encoder_input']
    decoder_input = data['decoder_input']
    decoder_expected_output = data['decoder_expected_output']

    # 合并数据
    np.savez_compressed(
        os.path.join(output_path, data_file),
        wav_filenames=wav_filenames,
        encoder_input=encoder_input,
        decoder_input=decoder_input,
        decoder_expected_output=decoder_expected_output
    )

    print(f'{os.path.join(output_path, data_file)}当中的文件已经合并')

print("所有文件已成功合并并存储。")
