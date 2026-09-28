# Purpose: Verify MediaPipePose and PoseFeatureExtractor can produce 46D features from test2.mp4.
# Run from the project root on RK3588 with required dependencies installed.

# ============================================================
# test_feature.py
#
# MediaPipe Pose + FeatureExtractor Test
#
# Output:
#   Feature vector:
#       (46,)
#
# ============================================================


import cv2
import numpy as np
import time


from npu.mediapipe_pose import MediaPipePose
from engine.feature import PoseFeatureExtractor



# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "test2.mp4"


TEST_INTERVAL = 5



# ============================================================
# INIT
# ============================================================


print("="*60)
print(" Feature Extractor Test ")
print("="*60)



# Pose

pose = MediaPipePose(
    model_complexity=1
)


# Feature

extractor = PoseFeatureExtractor()



# ============================================================
# VIDEO
# ============================================================


cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():

    raise RuntimeError(
        "Cannot open video:"
        + VIDEO_PATH
    )



frame_id = 0


success_count = 0

fail_count = 0



start=time.time()



# ============================================================
# LOOP
# ============================================================


while True:


    ret,frame=cap.read()


    if not ret:
        break



    frame_id += 1



    if frame_id % TEST_INTERVAL !=0:
        continue



    # ----------------------------------
    # Pose
    # ----------------------------------

    kpts = pose.detect(
        frame
    )



    print(
        "\nFRAME",
        frame_id
    )



    if kpts is None:


        print(
            "[Pose] failed"
        )


        fail_count +=1

        continue



    print(
        "Keypoints:",
        kpts.shape
    )



    # ----------------------------------
    # Feature
    # ----------------------------------

    feature = extractor.extract(
        kpts
    )



    if feature is None:


        print(
            "[Feature] failed"
        )


        fail_count +=1

        continue



    print(
        "Feature shape:",
        feature.shape
    )


    print(
        "dtype:",
        feature.dtype
    )



    print(
        "first 10:",
        feature[:10]
    )



    # check

    if feature.shape == (46,):


        print(
            "[Feature] OK"
        )


        success_count +=1



    else:


        print(
            "[Feature] ERROR"
        )


        fail_count +=1



# ============================================================
# SUMMARY
# ============================================================


cap.release()



cost=time.time()-start



print("\n")
print("="*60)
print(" TEST SUMMARY ")
print("="*60)


print(
    "Frames:",
    frame_id
)


print(
    "Success:",
    success_count
)


print(
    "Failed:",
    fail_count
)


print(
    "Time:",
    round(cost,2),
    "s"
)



if success_count>0:

    print(
        "\nFeature extractor is READY!"
    )

else:

    print(
        "\nFeature extractor FAILED!"
    )


print("="*60)
