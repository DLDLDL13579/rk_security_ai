# Purpose: Verify PoseRKNN can load npu/model.rknn and detect keypoints on person.jpg.
# Run from the project root on RK3588 with rknnlite and OpenCV installed.

from pathlib import Path

import cv2
import numpy as np

from npu.pose_rknn import PoseRKNN


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "npu" / "model.rknn"
IMAGE_PATH = ROOT / "person.jpg"


def main():
    print("=" * 60)
    print("Pose RKNN Smoke Test")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"pose model not found: {MODEL_PATH}")

    img = cv2.imread(str(IMAGE_PATH))
    if img is None:
        raise FileNotFoundError(f"image not found: {IMAGE_PATH}")

    print("input:", img.shape)
    pose = PoseRKNN(str(MODEL_PATH))
    kpts = pose.detect(img)

    print("type:", type(kpts))
    print("shape:", None if kpts is None else np.array(kpts).shape)
    print(kpts)
    pose.release()


if __name__ == "__main__":
    main()
