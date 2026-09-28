# ============================================================
# pose/mediapipe_pose.py
#
# MediaPipe Pose Wrapper
#
# Input:
#     person crop image
#
# Output:
#     33 landmarks
#
# Compatible with:
#     engine.feature.PoseFeatureExtractor
#
# ============================================================


import cv2
import mediapipe as mp


class MediaPipePose:


    def __init__(self, model_complexity=1):


        print("[Pose] MediaPipe Init")


        self.mp_pose = mp.solutions.pose


        self.pose = self.mp_pose.Pose(

            static_image_mode=False,

            model_complexity=model_complexity,

            smooth_landmarks=True,

            enable_segmentation=False,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5

        )


        print("[Pose] MediaPipe Ready")



    # =====================================================
    # Detect
    # =====================================================

    def detect(self, img):


        if img is None:

            return None



        if img.size == 0:

            return None



        try:


            rgb=cv2.cvtColor(

                img,

                cv2.COLOR_BGR2RGB

            )



            result=self.pose.process(rgb)



            if not result.pose_landmarks:


                return None



            landmarks=result.pose_landmarks.landmark



            return landmarks



        except Exception as e:


            print(

                "[MediaPipe ERROR]",

                e

            )


            return None



    # =====================================================
    # release
    # =====================================================

    def release(self):


        try:

            self.pose.close()

        except:

            pass
