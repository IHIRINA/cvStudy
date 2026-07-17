import os
import h5py  # 如需仅打印目录结构，可注释掉此导入及相关函数


# -------------------------- 1. 打印文件系统目录结构（匹配截图） --------------------------
def print_directory_structure(root_dir: str, indent: int = 0) -> None:
    """
    递归遍历目录，打印树状结构（和你截图的目录完全对应）
    :param root_dir: 数据根目录（即截图中的`data`文件夹路径）
    :param indent: 缩进层级，用于生成树状结构
    """
    # 分离目录和文件，保证先打印目录、再打印文件
    entries = os.listdir(root_dir)
    dirs = [e for e in entries if os.path.isdir(os.path.join(root_dir, e))]
    files = [e for e in entries if os.path.isfile(os.path.join(root_dir, e))]

    # 打印当前目录（根目录特殊处理）
    dir_name = os.path.basename(root_dir)
    print(f"{'  ' * indent}📂 {dir_name}")

    # 递归打印子目录
    for d in dirs:
        subdir_path = os.path.join(root_dir, d)
        print_directory_structure(subdir_path, indent + 1)

    # 打印当前目录下的文件
    for f in files:
        print(f"{'  ' * (indent + 1)}📄 {f}")


# -------------------------- 2. 打印HDF5文件内部数据结构 --------------------------
def print_single_hdf5_structure(hdf5_path: str, indent: int = 0) -> None:
    """
    打印单个HDF5文件的内部结构（组/数据集、形状、数据类型）
    :param hdf5_path: .hdf5文件的完整路径
    :param indent: 缩进层级
    """
    with h5py.File(hdf5_path, "r") as f:
        # 递归遍历HDF5的组和数据集
        def _visit_items(name: str, node: h5py.Group | h5py.Dataset) -> None:
            if isinstance(node, h5py.Group):
                print(f"{'  ' * indent}📦 Group: {name}")
            elif isinstance(node, h5py.Dataset):
                print(
                    f"{'  ' * indent}📊 Dataset: {name} | shape={node.shape} | dtype={node.dtype}"
                )

        f.visititems(_visit_items)


def print_all_hdf5_structure(root_dir: str) -> None:
    """
    遍历目录下所有.hdf5文件，批量打印内部数据结构
    :param root_dir: 数据根目录（`data`文件夹路径）
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".hdf5"):
                hdf5_full_path = os.path.join(root, file)
                print(f"\n=== HDF5文件: {hdf5_full_path} ===")
                print_single_hdf5_structure(hdf5_full_path)


# -------------------------- 3. 主函数：执行打印 --------------------------
if __name__ == "__main__":
    # 🔴 替换为你本地`data`文件夹的绝对路径！
    DATA_ROOT = "/root/autodl-tmp/data/hdf5"  # 示例："/home/your_name/data" 或 "D:/medical_data/data"

    # 1. 打印目录结构（和截图完全一致）
    print("=" * 50)
    print("📁 数据目录结构（匹配截图）")
    print("=" * 50)
    print_directory_structure(DATA_ROOT)

    # 2. 打印所有HDF5文件的内部数据结构（可选，注释后仅打印目录）
    print("\n" + "=" * 50)
    print("📊 所有HDF5文件内部数据结构")
    print("=" * 50)
    print_all_hdf5_structure(DATA_ROOT)
    