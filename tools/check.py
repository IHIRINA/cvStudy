import os
import nibabel as nib
import json

# 配置路径
json_path = "/root/autodl-tmp/data/latent_t1ce_tof_pairs.json"
data_root = "/root/autodl-tmp/data"

# 加载配对信息
with open(json_path) as f:
    data = json.load(f)["training"]

# 添加绝对路径
for item in data:
    item["image"] = os.path.join(data_root, item["image"])
    item["label"] = os.path.join(data_root, item["label"])

# 检查所有文件
corrupted = []
for idx, item in enumerate(data):
    for key in ["image", "label"]:
        fpath = item[key]
        try:
            img = nib.load(fpath)
            _ = img.get_fdata()  # 触发完整读取
        except Exception as e:
            corrupted.append((fpath, str(e)))
            print(f"[损坏] {fpath} -> {e}")

print(f"\n总计损坏文件数: {len(corrupted)}")
if corrupted:
    with open("corrupted_files.txt", "w") as f:
        for path, err in corrupted:
            f.write(f"{path}\t{err}\n")
    print("损坏文件列表已保存至 corrupted_files.txt")