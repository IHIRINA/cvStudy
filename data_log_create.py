'''
import os
import glob
import json

# 配置路径
image_dir = "/root/autodl-tmp/data/t1ce_latent"          # CBCT 目录
label_dir = "/root/autodl-tmp/data/tof_latent"  # CT 目录
output_json_path = "/root/autodl-tmp/data/latent_t1ce_tof_pairs.json"
sim_dim = [64, 64,32]

# 仅保留 .nii.gz 文件基名
cbct_files = {f for f in os.listdir(image_dir) if f.endswith('.nii.gz')}
ct_files   = {f for f in os.listdir(label_dir)  if f.endswith('.nii.gz')}

# 取交集：同名文件
paired_files = sorted(cbct_files & ct_files)

datalist = {"training": [], "validation": []}

for idx, fname in enumerate(paired_files):
    cbct_path = os.path.join(image_dir, fname)
    ct_path   = os.path.join(label_dir, fname)

    datalist["training"].append({
        "image": cbct_path,      # CBCT
        "label": ct_path,        # CT
        "fold": idx % 10,        # 10 折
        "dim": sim_dim,
        "spacing": [1.0, 1.0, 2.0],
        "case_id": fname.replace('.nii.gz', '')
    })

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(datalist, f, indent=2, ensure_ascii=False)

print(f"JSON文件已生成: {output_json_path}")
print(f"共找到 {len(datalist['training'])} 个同名CT-CBCT配对")
'''

import os
import re
import json
from collections import defaultdict

# 配置路径
t1ce_dir = "/root/autodl-tmp/data/T1ce_latent"   # T1CE 图像目录
tof_dir  = "/root/autodl-tmp/data/TOF_latent"    # TOF 图像目录
output_json_path = "/root/autodl-tmp/data/latent_t1ce_tof_pairs.json"
sim_dim = [64, 64, 32]

def extract_id(filename):
    """
    从文件名中提取配对ID。
    例如：
        "1-1001398096-TOF_pre.nii.gz"      -> "1-1001398096"
        "1-1001398096-11_pre_reg2TOF aladin.nii.gz" -> "1-1001398096"
    通过按 '-' 分割，取前两部分用 '-' 连接得到 ID。
    """
    parts = filename.split('-', 2)  # 最多分割两次
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    else:
        return None

# 获取两个目录中所有 .nii.gz 文件
t1ce_files = {f for f in os.listdir(t1ce_dir) if f.endswith('.nii.gz')}
tof_files  = {f for f in os.listdir(tof_dir)  if f.endswith('.nii.gz')}

# 构建 ID -> 文件路径 的映射
t1ce_dict = {}
for fname in t1ce_files:
    fid = extract_id(fname)
    if fid:
        t1ce_dict[fid] = os.path.join(t1ce_dir, fname)

tof_dict = {}
for fname in tof_files:
    fid = extract_id(fname)
    if fid:
        tof_dict[fid] = os.path.join(tof_dir, fname)

# 取共同 ID
common_ids = sorted(set(t1ce_dict.keys()) & set(tof_dict.keys()))
print(f"找到 {len(common_ids)} 个可配对的 ID")

# 生成数据列表
datalist = {"training": [], "validation": []}
for idx, fid in enumerate(common_ids):
    datalist["training"].append({
        "image": t1ce_dict[fid],    # T1CE 作为图像
        "label": tof_dict[fid],     # TOF  作为标签
        "fold": idx % 10,
        "dim": sim_dim,
        "spacing": [1.0, 1.0, 2.0],
        "case_id": fid
    })

# 保存 JSON
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(datalist, f, indent=2, ensure_ascii=False)

print(f"配对 JSON 文件已生成: {output_json_path}")
print(f"训练集配对个数: {len(datalist['training'])}")