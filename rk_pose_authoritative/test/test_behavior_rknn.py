import numpy as np
from pathlib import Path

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

result = behavior.infer(
    input_data
)



print("\nResult")
print("----------------------")


print(
    "Class:",
    result["class_name"]
)


print(
    "ID:",
    result["class_id"]
)


print(
    "Confidence:",
    result["confidence"]
)


print("======================")
print("TEST DONE")
print("======================")


behavior.release()
