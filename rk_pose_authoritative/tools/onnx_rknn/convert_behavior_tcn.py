# -*- coding: utf-8 -*-
"""
============================================================
onnx_rknn/convert_behavior_tcn.py

行为 TCN 模型 ONNX -> RKNN 转换（目标平台 RK3588）

ONNX 约定:
    input  : [1, 16, 46]   (batch, seq_len, feature_dim)
    output : [1, 5]        (standing walking squat bend fall_down)

关键设计:
  - 转换前用 onnx 包统一准备模型:
      1) opset 超过 15 -> 降级为 15
      2) 输入含动态维度 -> 重写为固定 [1,16,46]
    这样 load_onnx 不再传 inputs/input_size_list，不触发 rknn 的
    crop 模式（rknn 2.3.2 crop 模式会把模型重存成 opset 19 导致 build 失败）。

用法（在装有 rknn-toolkit2 的 PC 上执行）:
    python convert_behavior_tcn.py --onnx behavior_tcn.onnx --calib-dir calib --verify
    python convert_behavior_tcn.py --no-quantize
============================================================
"""

import argparse
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from rknn.api import RKNN


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_ONNX = SCRIPT_DIR / "behavior_tcn.onnx"
DEFAULT_OUTPUT = SCRIPT_DIR / "behavior_tcn.rknn"
DEPLOY_TARGET = PROJECT_ROOT / "models" / "behavior_tcn.rknn"

INPUT_SHAPE = (1, 16, 46)
CLASSES = ["standing", "walking", "squat", "bend", "fall_down"]


# ============================================================
# ONNX 准备：opset 降级 + 动态维度固定（避免 rknn crop 模式）
# ============================================================

def prepare_onnx(onnx_path):
    """返回可直接加载的 onnx 路径。

    1) 打印并检查 opset，>15 降级为 15；
    2) 若输入含动态维度，重写为固定 [1,16,46]。
    任一改动都会另存为 behavior_tcn_prepared.onnx。
    """
    try:
        import onnx
    except ImportError:
        raise RuntimeError("需要 onnx 包: pip install onnx")

    model = onnx.load(str(onnx_path))

    # ---- opset ----
    versions = [op.version for op in model.opset_import]
    opset = max(versions) if versions else 1
    print("[INFO] ONNX opset:", opset,
          "domains:", [(op.domain, op.version) for op in model.opset_import])
    opset_changed = False
    if opset > 15:
        print(f"[FIX] opset {opset} > 15，降级为 15 ...")
        for op in model.opset_import:
            if op.domain in ("", "ai.onnx"):
                op.version = 15
        opset_changed = True
    else:
        print("[INFO] opset <= 15，无需降级")

    # ---- 动态维度固定 ----
    inp = model.graph.input[0]
    dims = inp.type.tensor_type.shape.dim
    desc = []
    shape_changed = False
    for i, d in enumerate(dims):
        if d.HasField("dim_param"):
            d.ClearField("dim_param")
            d.dim_value = INPUT_SHAPE[i]
            shape_changed = True
            desc.append(str(INPUT_SHAPE[i]))
        else:
            desc.append(str(d.dim_value))
    print("[INFO] input dims:", desc, "| dynamic:", shape_changed)

    if opset_changed or shape_changed:
        fixed_path = SCRIPT_DIR / "behavior_tcn_prepared.onnx"
        onnx.save(model, str(fixed_path))
        print("[FIX] 已保存准备后的模型:", fixed_path)
        return fixed_path

    print("[INFO] 无需修改，直接使用原文件")
    return onnx_path


# ============================================================
# 校准数据
# ============================================================

def collect_calibration(args):
    """返回 (dataset_txt_path, tmp_dir) 或 (None, None)。"""
    if args.dataset:
        txt = Path(args.dataset)
        if not txt.exists():
            raise FileNotFoundError(f"dataset file not found: {txt}")
        return str(txt), None

    tmp_dir = tempfile.mkdtemp(prefix="rknn_calib_")
    txt_path = os.path.join(tmp_dir, "calib.txt")

    if args.calib_dir:
        calib_dir = Path(args.calib_dir)
        if not calib_dir.exists():
            raise FileNotFoundError(f"calib dir not found: {calib_dir}")

        # rknn 的 dataset 解析不支持路径含空格（如 rk_pose_project _8.7），
        # 统一把校准样本复制到无空格的临时目录再写 calib.txt
        calib_tmp = Path(tmp_dir) / "data"
        calib_tmp.mkdir(parents=True, exist_ok=True)
        picked = []
        for cls in CLASSES:
            folder = calib_dir / cls
            if not folder.is_dir():
                continue
            files = sorted(folder.glob("*.npy"))
            # 按视频分组均匀采样，避免校准集只来自同一视频开头
            groups = {}
            for f in files:
                parts = f.stem.split("_")
                key = parts[1] if len(parts) >= 3 and parts[1].isdigit() else "legacy"
                groups.setdefault(key, []).append(f)
            n_groups = max(1, len(groups))
            per_video = max(1, int(math.ceil(args.calib_per_class / n_groups)))
            for key in sorted(groups):
                for f in sorted(groups[key])[:per_video]:
                    # rknn 量化输入需要 3 维 (1,16,46)，训练样本是 (16,46)
                    arr = np.load(f)
                    if arr.ndim == 2:
                        arr = np.expand_dims(arr, axis=0)
                    dst = calib_tmp / f"{cls}_{key}_{f.name}"
                    np.save(dst, arr.astype(np.float32))
                    picked.append(str(dst))
        if not picked:
            raise RuntimeError(f"calib dir 下未找到 npy: {calib_dir}")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(picked) + "\n")
        print(f"[CALIB] 使用真实训练数据 {len(picked)} 个样本"
              f"（每类最多 {args.calib_per_class}）")
        print("[CALIB] 临时目录（无空格）:", tmp_dir)
        return txt_path, tmp_dir

    print("[WARN] 未提供 --dataset / --calib-dir，将用随机数据校准，"
          "量化精度可能下降")
    for i in range(50):
        np.save(
            os.path.join(tmp_dir, f"calib_{i:03d}.npy"),
            np.random.randn(16, 46).astype(np.float32),
        )
    with open(txt_path, "w", encoding="utf-8") as f:
        for i in range(50):
            f.write(os.path.join(tmp_dir, f"calib_{i:03d}.npy") + "\n")
    return txt_path, tmp_dir


# ============================================================
# 转换
# ============================================================

def convert(args):
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX 模型不存在: {onnx_path}\n"
            "请先在训练工程运行 pipelines/export_onnx.py 导出，"
            "并把 behavior_tcn.onnx 放到本目录"
        )

    onnx_path = prepare_onnx(onnx_path)
    dataset_txt, tmp_dir = collect_calibration(args)

    rknn = RKNN(verbose=args.verbose)

    try:
        try:
            import onnx as _onnx
            import onnxruntime as _ort
            print("[INFO] onnx:", _onnx.__version__,
                  "| onnxruntime:", _ort.__version__)
        except Exception:
            pass

        print("[1/4] config ...")
        rknn.config(
            target_platform=args.target_platform,
            quantized_dtype=(
                "float16"
                if args.no_quantize
                else "asymmetric_quantized-8"
            ),
        )

        print("[2/4] load onnx ...", onnx_path)
        print("[INFO] direct load（不传 inputs，无 crop 模式）")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError("load_onnx failed")

        print("[3/4] build ...")
        try:
            if args.no_quantize:
                ret = rknn.build(do_quantization=False)
            else:
                ret = rknn.build(
                    do_quantization=True,
                    dataset=dataset_txt,
                )
        except Exception as e:
            if "Opset" in str(e) or "opset" in str(e):
                print("[HINT] 这是 rknn 环境的 onnx/onnxruntime 版本组合问题：")
                print("[HINT] rknn 构建时内部重写模型为 opset 19，而 onnxruntime 1.10 只支持到 15。")
                print("[HINT] 修复（在 rknn 环境里按顺序执行）：")
                print("[HINT]   1) pip install onnxruntime==1.16.3   # 让 onnxruntime 支持 opset 19")
                print("[HINT]   2) 仍失败再执行: pip install onnx==1.14.1   # 对齐 rknn-toolkit2 2.3.x 配套")
                print("[HINT]   3) 仍失败: 换 rknn-toolkit2 版本（如 1.6.0）")
            raise
        if ret != 0:
            raise RuntimeError("build failed")

        print("[4/4] export ...", args.output)
        ret = rknn.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError("export_rknn failed")
        print("[DONE] saved:", args.output)

        if args.verify:
            print("[VERIFY] 若此步出现 malloc/Aborted 崩溃：是 rknn 模拟器与")
            print("[VERIFY] onnxruntime 1.16.3 的兼容问题，模型已导出成功，")
            print("[VERIFY] 可直接去掉 --verify 使用 behavior_tcn.rknn")
            verify(rknn, onnx_path, args.output)

    finally:
        rknn.release()
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# 验证：ONNX vs RKNN 输出对比
# ============================================================

def verify(rknn, onnx_path, rknn_path):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[VERIFY] 未安装 onnxruntime，跳过对比")
        return

    sess = ort.InferenceSession(str(onnx_path))
    rng = np.random.RandomState(0)
    x = rng.randn(*INPUT_SHAPE).astype(np.float32)

    onnx_out = sess.run(None, {"input": x})[0]

    ret = rknn.init_runtime()
    if ret != 0:
        print("[VERIFY] init_runtime failed，跳过")
        return
    rknn_out = rknn.inference(inputs=[x])[0]

    def show(name, out):
        pred = int(np.argmax(out))
        print(f"  {name}: {np.round(out, 4)} -> {CLASSES[pred]}")

    show("ONNX", onnx_out[0])
    show("RKNN", np.asarray(rknn_out)[0])
    diff = float(np.abs(onnx_out - np.asarray(rknn_out)).max())
    print("  max abs diff:", round(diff, 6))
    onnx_pred = int(np.argmax(onnx_out[0]))
    rknn_pred = int(np.argmax(np.asarray(rknn_out)[0]))
    same = onnx_pred == rknn_pred
    print(f"  argmax: ONNX={CLASSES[onnx_pred]} vs RKNN={CLASSES[rknn_pred]}"
          f"（{'一致' if same else '不一致'}）")
    if diff < 0.05 and same:
        print("[VERIFY] 转换正确")
    else:
        print("[VERIFY] 差异偏大：量化输出为 int8 且随机输入不代表真实特征；")
        print("[VERIFY] 判断标准以上板实测为准。若真实视频上明显掉点：")
        print("[VERIFY]   1) 加多校准样本 --calib-per-class 50")
        print("[VERIFY]   2) 或转 float16 --no-quantize 对比")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Behavior TCN ONNX -> RKNN (RK3588)"
    )
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX),
                        help=f"ONNX 路径（默认 {DEFAULT_ONNX}）")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help=f"RKNN 输出路径（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--dataset", default=None,
                        help="校准数据 txt（每行一个 .npy 绝对路径）")
    parser.add_argument("--calib-dir", default=None,
                        help="训练数据集目录，自动每类取样做校准")
    parser.add_argument("--calib-per-class", type=int, default=20,
                        help="每类校准样本数（默认 20）")
    parser.add_argument("--no-quantize", action="store_true",
                        help="跳过 INT8 量化（float16）")
    parser.add_argument("--target-platform", default="rk3588",
                        help="目标平台（默认 rk3588）")
    parser.add_argument("--verify", action="store_true",
                        help="转换后用 onnxruntime 与 rknn 对比一次输出")
    parser.add_argument("--verbose", action="store_true",
                        help="打开 RKNN 详细日志")
    args = parser.parse_args()

    convert(args)

    print()
    print("下一步:")
    print(f"  拷贝到部署位置: cp {args.output} {DEPLOY_TARGET}")
    print("  板端验证: python test_multi_behavior_eval.py --source videos")


if __name__ == "__main__":
    main()
