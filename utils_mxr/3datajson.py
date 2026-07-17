'''
import h5py
root = '/root/autodl-tmp/data/hdf5/hdf5_data'
subject = '1000454786'  # replace with one of your actual IDs
modal = '3D-FLAIR'
with h5py.File(f'{root}/{subject}/{modal}.hdf5', 'r') as f:
    print(f.keys())  # should print 'data', 'max', 'max_len'
    '''

import os
import random
import json

data_root = '/root/autodl-tmp/data/hdf5/hdf5_data/'
# 获取所有 subject 文件夹名称
subjects = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

# 设置随机种子，保证结果可复现
random.seed(42)
random.shuffle(subjects)

# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

n_total = len(subjects)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
# 剩余的作为测试集
n_test = n_total - n_train - n_val

train_list = subjects[:n_train]
val_list = subjects[n_train:n_train + n_val]
test_list = subjects[n_train + n_val:]

# 保存到 JSON 文件（可以直接覆盖原来的 3data.json）
with open('32data.json', 'w') as f:
    json.dump({'train': train_list, 'valid': val_list, 'test': test_list}, f, indent=2)

print(f"训练集: {len(train_list)} 个")
print(f"验证集: {len(val_list)} 个")
print(f"测试集: {len(test_list)} 个")