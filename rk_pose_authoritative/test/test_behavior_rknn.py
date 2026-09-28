import os
import sys

import numpy as np
from pathlib import Path

# 工程根目录加入 sys.path，保证直接运行本脚本时能 import 到 npu/engine
# （2026-09-28 修正：原脚本缺此处理，直接运行会 ModuleNotFoundError: No module named 'npu'）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from npu.behavior_rknn import BehaviorRKNN



print("="*60)
print("Behavior RKNN Test")
print("="*60)


ROOT = Path(__file__).resolve().parent.parent


MODEL_PATH = (
    ROOT /
    "models/"
    "behavior_tcn.rknn"
)


print("Model:")
print(MODEL_PATH)



# ======================
# Load
# ======================

behavior = BehaviorRKNN(
    str(MODEL_PATH)
)


print("[OK] Behavior model loaded")



# ======================
# fake input
# ======================

input_data = np.random.randn(
    16,
    46
).astype(
    np.float32
)


print("\nInput:")
print(input_data.shape)



# ======================
# inference
# ======================

# BehaviorRKNN.forward() 返回元组 (label: str, confidence: float)
# 原脚本调用不存在的 behavior.infer() 并按 dict 解析，2026-09-28 修正
label, confidence = behavior.forward(
    input_data
)


print("\nResult")
print("----------------------")

print(
    "Class:",
    label
)

print(
    "Confidence:",
    confidence
)


print("======================")
print("TEST DONE")
print("======================")


behavior.release()
