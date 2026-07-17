#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ----------------------------
# I/O & ID mapping
# ----------------------------
def iter_nii(input_dir: str) -> Iterable[str]:
    p = Path(input_dir)
    for ext in ("*.nii.gz", "*.nii"):
        yield from (str(x) for x in p.rglob(ext))


def patient_id_from_name(path: str) -> str:
    """
    例：
      IXI002-Guys-0828-MRA.nii.gz -> IXI002-Guys-0828
      IXI002-Guys-0828-T1.nii.gz  -> IXI002-Guys-0828
    """
    name = os.path.basename(path)
    base = name.replace(".nii.gz", "").replace(".nii", "")
    parts = base.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:-1])
    return base


def build_id_map(input_dir: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    for p in iter_nii(input_dir):
        pid = patient_id_from_name(p)
        mp[pid] = p
    return mp


# ----------------------------
# spacing / resampling
# ----------------------------
def get_spacing(path: str) -> np.ndarray:
    img = sitk.ReadImage(path)
    return np.array(img.GetSpacing(), dtype=np.float64)


def median_spacing(paths: List[str]) -> np.ndarray:
    spacings = np.stack([get_spacing(p) for p in paths], axis=0)
    return np.median(spacings, axis=0)


def _img_stats(tag: str, img: sitk.Image) -> str:
    sz = img.GetSize()  # (x,y,z)
    sp = img.GetSpacing()
    vox = float(sz[0] * sz[1] * sz[2]) / 1e6
    return f"[{tag}] size={sz} spacing=({sp[0]:.4f},{sp[1]:.4f},{sp[2]:.4f}) vox={vox:.1f}M"


def resample_to_spacing(
    img: sitk.Image,
    target_spacing: np.ndarray,
    *,
    interp: str = "linear",
) -> sitk.Image:
    """
    将 img 重采样到 target_spacing（保持原方向/原点），输出 size 按比例缩放
    """
    target_spacing = [float(x) for x in target_spacing]
    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()
    new_size = [
        int(np.round(orig_size[i] * (orig_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]
    new_size = [max(1, s) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if interp == "nearest":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interp == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    # 输出强制 float32，避免外部工具不兼容 int16/uint 等
    out = resampler.Execute(sitk.Cast(img, sitk.sitkFloat32))
    return out


# ----------------------------
# reg_aladin (CPU)
# ----------------------------
def run_cmd(cmd: List[str], *, verbose: bool = False) -> None:
    if verbose:
        print("[CMD]", " ".join(cmd))
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"[CMD FAILED]\nCMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{r.stdout}\n"
            f"STDERR:\n{r.stderr}\n"
        )
    if verbose and r.stdout.strip():
        print("[STDOUT]\n" + r.stdout.strip())
    # 有些版本把信息打印在 stderr
    if verbose and r.stderr.strip():
        print("[STDERR]\n" + r.stderr.strip())


def niftyreg_register_aladin_cpu(
    fixed_img: sitk.Image,
    moving_img: sitk.Image,
    *,
    tmp_dir: str,
    reg_aladin_path: str = "reg_aladin",
    dof: str = "affine",   # "rigid" or "affine"
    omp: int = 4,
    verbose: bool = False,
) -> sitk.Image:
    """
    moving -> fixed，输出在 fixed 空间中的 moving_res（CPU 版 reg_aladin）
    提速点：临时文件用 .nii 且不压缩
    """
    if not (os.path.isabs(reg_aladin_path) and os.path.exists(reg_aladin_path)):
        import shutil as _shutil
        if _shutil.which(reg_aladin_path) is None:
            raise FileNotFoundError(
                f"Cannot find '{reg_aladin_path}' in PATH. "
                f"Try --reg_aladin /full/path/to/reg_aladin"
            )

    os.makedirs(tmp_dir, exist_ok=True)
    fixed_path = os.path.join(tmp_dir, "fixed.nii")
    moving_path = os.path.join(tmp_dir, "moving.nii")
    out_res_path = os.path.join(tmp_dir, "moving_reg.nii")
    out_aff_path = os.path.join(tmp_dir, "aladin_aff.txt")

    fixed_f = sitk.Cast(fixed_img, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving_img, sitk.sitkFloat32)

    # 不压缩写入：快很多
    sitk.WriteImage(fixed_f, fixed_path, useCompression=False)
    sitk.WriteImage(moving_f, moving_path, useCompression=False)

    cmd = [
        reg_aladin_path,
        "-ref", fixed_path,
        "-flo", moving_path,
        "-res", out_res_path,
        "-aff", out_aff_path,
        "-omp", str(int(omp)),
    ]
    if dof == "rigid":
        cmd += ["-rigOnly"]

    run_cmd(cmd, verbose=verbose)

    if not os.path.exists(out_res_path):
        raise RuntimeError("reg_aladin did not produce output resampled image.")
    return sitk.ReadImage(out_res_path)


# ----------------------------
# crop / normalize
# ----------------------------
def bbox_from_mask(mask_arr_zyx: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    idx = np.argwhere(mask_arr_zyx)
    if idx.size == 0:
        return None
    z0, y0, x0 = idx.min(axis=0)
    z1, y1, x1 = idx.max(axis=0) + 1
    return (int(z0), int(y0), int(x0), int(z1), int(y1), int(x1))


def crop_with_sitk_roi(img: sitk.Image, bbox_zyx, margin: int = 5) -> sitk.Image:
    """
    bbox_zyx: (z0, y0, x0, z1, y1, x1)  (z1/y1/x1 为开区间)
    SITK ROI 需要 (x,y,z) 顺序的 index/size，且类型必须是 Python int (unsigned)。
    """
    z0, y0, x0, z1, y1, x1 = bbox_zyx
    size_xyz = img.GetSize()  # (x,y,z), tuple of python ints

    # margin + clamp
    x0_ = max(0, int(x0) - int(margin))
    y0_ = max(0, int(y0) - int(margin))
    z0_ = max(0, int(z0) - int(margin))

    x1_ = min(int(size_xyz[0]), int(x1) + int(margin))
    y1_ = min(int(size_xyz[1]), int(y1) + int(margin))
    z1_ = min(int(size_xyz[2]), int(z1) + int(margin))

    # size must be positive
    sx = max(1, x1_ - x0_)
    sy = max(1, y1_ - y0_)
    sz = max(1, z1_ - z0_)

    roi_index = [int(x0_), int(y0_), int(z0_)]
    roi_size  = [int(sx),  int(sy),  int(sz)]

    return sitk.RegionOfInterest(img, size=roi_size, index=roi_index)


def zscore_normalize(img: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (z,y,x)
    m = arr != 0
    if int(m.sum()) < 10:
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        return out
    vals = arr[m]
    mu = float(vals.mean())
    sd = float(vals.std()) + 1e-8
    arr = (arr - mu) / sd
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out

# 新增变成32倍数的尺寸
def pad_to_target_size(img: sitk.Image, target_size: Tuple[int, int, int] = (224, 224, 160)) -> sitk.Image:
    """居中 padding 到固定尺寸（推荐用于 3D 生成任务）"""
    size = img.GetSize()  # (x, y, z)
    if list(size) == list(target_size):
        return img

    lower = [(target_size[i] - size[i]) // 2 for i in range(3)]
    upper = [target_size[i] - size[i] - lower[i] for i in range(3)]

    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound(lower)
    pad_filter.SetPadUpperBound(upper)
    pad_filter.SetConstant(0.0)   # zscore 后背景为 0

    return pad_filter.Execute(img)



# ----------------------------
# main
# ----------------------------
def _set_sitk_threads(n: int):
    # 不同 SimpleITK 版本 API 名字略有差异，做个兼容
    try:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(n))
    except Exception:
        try:
            sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(int(n))  # type: ignore
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1_dir", type=str, default='/root/autodl-tmp/T1')
    ap.add_argument("--tof_dir", type=str, default='/root/autodl-tmp/TOF')
    ap.add_argument("--out_dir", type=str, default='/root/autodl-tmp/output')

    # spacing & resample
    ap.add_argument("--min_spacing", type=float, nargs=3, default=[0.8, 0.8, 0.8],
                    help="target spacing 下限，防止 spacing 太小导致体素数暴涨")
    ap.add_argument("--resample_interp", type=str, default="linear",
                    choices=["linear", "bspline"], help="重采样插值（推荐 linear 更快）")

    # crop & normalize
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--target_size", type=int, nargs=3, default=[224, 224, 160],
                    help="最终输出图像尺寸 (x y z)，默认 224 224 160")

    # reg_aladin CPU
    ap.add_argument("--reg_aladin", type=str, default="/root/autodl-tmp/brain-mri-preprocess/preprocess/niftyreg/build/reg-apps/reg_aladin",
                    help="reg_aladin 可执行文件名或绝对路径")
    ap.add_argument("--aladin_dof", type=str, default="affine", choices=["rigid", "affine"])
    ap.add_argument("--aladin_omp", type=int, default=8)

    # threads env
    ap.add_argument("--set_omp_env", action="store_true",
                    help="设置 OMP_NUM_THREADS/OMP_PROC_BIND/OMP_PLACES")
    ap.add_argument("--sitk_threads", type=int, default=1,
                    help="限制 SimpleITK 线程数，避免与 reg_aladin 抢线程（推荐 1）")

    # tmp / debug
    ap.add_argument("--tmp_dir", type=str, default="", help="临时目录（默认 out_dir/_tmp）")
    ap.add_argument("--keep_tmp", action="store_true")
    ap.add_argument("--no_gz", action="store_true",
                    help="最终输出也不压缩（更快），输出 .nii 而不是 .nii.gz")
    ap.add_argument("--debug_time", action="store_true", help="打印每步耗时与体素规模")
    ap.add_argument("--verbose_cmd", action="store_true", help="打印 reg_aladin 的 stdout/stderr")
    args = ap.parse_args()

    if args.set_omp_env:
        os.environ["OMP_NUM_THREADS"] = str(int(args.aladin_omp))
        os.environ["OMP_PROC_BIND"] = "true"
        os.environ["OMP_PLACES"] = "cores"

    _set_sitk_threads(args.sitk_threads)

    t1_map = build_id_map(args.t1_dir)
    tof_map = build_id_map(args.tof_dir)
    ids = sorted(set(t1_map.keys()) & set(tof_map.keys()))
    if len(ids) == 0:
        raise RuntimeError("No matched patient ids between T1 and TOF. Check naming & patient_id_from_name().")

    os.makedirs(args.out_dir, exist_ok=True)

    # target spacing = median(TOF) then clamp to min_spacing
    tof_paths = [tof_map[i] for i in ids]
    tgt_spacing = median_spacing(tof_paths)
    min_sp = np.array(args.min_spacing, dtype=np.float64)
    tgt_spacing = np.maximum(tgt_spacing, min_sp)

    print(f"[INFO] Target spacing (median TOF, clamped) = {tgt_spacing.tolist()}")
    print(f"[INFO] resample_interp={args.resample_interp} | aladin dof={args.aladin_dof} | -omp {args.aladin_omp} | sitk_threads={args.sitk_threads}")
    if args.no_gz:
        print("[INFO] Output compression: OFF (.nii)")
    else:
        print("[INFO] Output compression: ON (.nii.gz)")

    tmp_root = args.tmp_dir if args.tmp_dir else os.path.join(args.out_dir, "_tmp")
    os.makedirs(tmp_root, exist_ok=True)

    for pid in tqdm(ids, desc="Pipeline (resample->reg->crop->norm->save)"):
        t_case0 = time.perf_counter()

        t1_path = t1_map[pid]
        tof_path = tof_map[pid]

        # 1) read
        t = time.perf_counter()
        t1 = sitk.ReadImage(t1_path)
        tof = sitk.ReadImage(tof_path)
        if args.debug_time:
            print(_img_stats("T1 raw", t1))
            print(_img_stats("TOF raw", tof))
            print(f"[TIMER] read: {time.perf_counter()-t:.2f}s")

        # 2) resample to target spacing
        t = time.perf_counter()
        tof_r = resample_to_spacing(tof, tgt_spacing, interp=args.resample_interp)
        t1_r = resample_to_spacing(t1, tgt_spacing, interp=args.resample_interp)
        if args.debug_time:
            print(_img_stats("TOF rs", tof_r))
            print(_img_stats("T1 rs", t1_r))
            print(f"[TIMER] resample: {time.perf_counter()-t:.2f}s")

        # 3) register T1 -> TOF
        t = time.perf_counter()
        case_tmp = os.path.join(tmp_root, pid)
        t1_reg = niftyreg_register_aladin_cpu(
            fixed_img=tof_r,
            moving_img=t1_r,
            tmp_dir=case_tmp,
            reg_aladin_path=args.reg_aladin,
            dof=args.aladin_dof,
            omp=args.aladin_omp,
            verbose=args.verbose_cmd,
        )
        if args.debug_time:
            print(_img_stats("T1 reg", t1_reg))
            print(f"[TIMER] reg_aladin: {time.perf_counter()-t:.2f}s")

        # 4) crop (use TOF nonzero bbox)
        t = time.perf_counter()
        tof_arr = sitk.GetArrayFromImage(tof_r)
        bbox = bbox_from_mask(tof_arr != 0)
        if bbox is not None:
            tof_c = crop_with_sitk_roi(tof_r, bbox, margin=args.margin)
            t1_c = crop_with_sitk_roi(t1_reg, bbox, margin=args.margin)
        else:
            tof_c, t1_c = tof_r, t1_reg
        if args.debug_time:
            print(_img_stats("TOF crop", tof_c))
            print(_img_stats("T1 crop", t1_c))
            print(f"[TIMER] crop: {time.perf_counter()-t:.2f}s")

        # 5) intensity normalize
        t = time.perf_counter()
        tof_n = zscore_normalize(tof_c)
        t1_n = zscore_normalize(t1_c)
        if args.debug_time:
            print(f"[TIMER] zscore: {time.perf_counter()-t:.2f}s")

        # 新增 6) pad to target size
        t = time.perf_counter()
        tof_n = pad_to_target_size(tof_n, tuple(args.target_size))
        t1_n = pad_to_target_size(t1_n, tuple(args.target_size))
        if args.debug_time:
            print(_img_stats("TOF pad", tof_n))
            print(_img_stats("T1 pad", t1_n))
            print(f"[TIMER] pad: {time.perf_counter()-t:.2f}s")


        # 6) save
        t = time.perf_counter()
        out_case = os.path.join(args.out_dir, pid)
        os.makedirs(out_case, exist_ok=True)

        if args.no_gz:
            tof_out = os.path.join(out_case, f"{pid}-TOF_pre.nii")
            t1_out = os.path.join(out_case, f"{pid}-T1_pre_reg2TOF_aladin.nii")
            sitk.WriteImage(tof_n, tof_out, useCompression=False)
            sitk.WriteImage(t1_n, t1_out, useCompression=False)
        else:
            tof_out = os.path.join(out_case, f"{pid}-TOF_pre.nii.gz")
            t1_out = os.path.join(out_case, f"{pid}-T1_pre_reg2TOF_aladin.nii.gz")
            sitk.WriteImage(tof_n, tof_out, useCompression=True)
            sitk.WriteImage(t1_n, t1_out, useCompression=True)

        if args.debug_time:
            print(f"[TIMER] write: {time.perf_counter()-t:.2f}s")

        # cleanup tmp
        if (not args.keep_tmp) and os.path.isdir(case_tmp):
            shutil.rmtree(case_tmp, ignore_errors=True)

        if args.debug_time:
            print(f"[TIMER] case total: {time.perf_counter()-t_case0:.2f}s\n")

    print(f"[OK] Done. Output: {args.out_dir}")
    print(f"[INFO] Temp root: {tmp_root} (keep_tmp={args.keep_tmp})")


if __name__ == "__main__":
    main()
