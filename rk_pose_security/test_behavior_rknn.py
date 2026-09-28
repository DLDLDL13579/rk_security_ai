# Purpose: Verify BehaviorRKNN can load behavior_tcn.rknn and run one fake (16, 46) sequence.
# Run from the project root on RK3588 with rknnlite and numpy installed.

import numpy as np
from pathlib import Path

from npu.behavior_rknn import BehaviorRKNN


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "npu" / "onnx-rknn" / "behavior_tcn.rknn"


def main():
    print("=" * 60)
    print("Behavior RKNN Forward Test")
    print("=" * 60)
    print("Model:", MODEL_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"behavior model not found: {MODEL_PATH}")

    behavior = BehaviorRKNN(str(MODEL_PATH))

    input_data = np.random.randn(16, 46).astype(np.float32)
    print("Input:", input_data.shape, input_data.dtype)

    label, confidence = behavior.forward(input_data)
    print("Result:")
    print("  label:", label)
    print("  confidence:", confidence)

    behavior.release()
    print("TEST DONE")


if __name__ == "__main__":
    main()
