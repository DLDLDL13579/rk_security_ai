import cv2
import numpy as np
import time
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from npu.pose_rknn import PoseRKNN



# ==========================
# 配置
# ==========================

MODEL_PATH = os.environ.get(
    "RK_POSE_MODEL",
    os.path.join(PROJECT_ROOT, "models", "yolov8n-pose.rknn"),
)

IMAGE_PATH = os.path.join(PROJECT_ROOT, "test", "person.jpg")

SAVE_PATH = os.path.join(PROJECT_ROOT, "output", "pose_test.jpg")



# ==========================
# COCO17骨架
# ==========================

# MediaPipe 33 点骨架连线表（detect() 经 convert17to33 返回 MediaPipe 兼容格式）
# 索引对照 MediaPipe Pose：11=左肩 12=右肩 13=左肘 14=右肘 15=左腕 16=右腕
# 23=左髋 24=右髋 25=左膝 26=右膝 27=左踝 28=右踝
# 注：convert17to33 只填充 COCO 17 点对应的位置，面部/手部点为占位值
SKELETON = [

    (11,12),   # 肩-肩

    (11,13),
    (13,15),   # 左臂

    (12,14),
    (14,16),   # 右臂

    (11,23),
    (12,24),   # 肩-髋

    (23,24),   # 髋-髋

    (23,25),
    (25,27),   # 左腿

    (24,26),
    (26,28)    # 右腿

]



def draw_pose(
        img,
        kpts
):

    # detect() 返回归一化坐标(0~1)，需乘以图像宽高换算成像素
    h, w = img.shape[:2]

    for x,y,score in kpts:


        if score > 0.2:


            cv2.circle(

                img,

                (
                    int(x * w),
                    int(y * h)
                ),

                5,

                (0,255,0),

                -1
            )



    for a,b in SKELETON:


        if (
            kpts[a][2]>0.2
            and
            kpts[b][2]>0.2
        ):


            cv2.line(

                img,

                (
                    int(kpts[a][0] * w),
                    int(kpts[a][1] * h)
                ),

                (
                    int(kpts[b][0] * w),
                    int(kpts[b][1] * h)
                ),

                (255,0,0),

                2
            )


    return img




def main():


    print("="*60)
    print("POSE RKNN SINGLE TEST")
    print("="*60)



    # ----------------------
    # image
    # ----------------------

    img=cv2.imread(
        IMAGE_PATH
    )


    if img is None:

        raise RuntimeError(
            "image not found"
        )


    print(
        "[IMAGE]",
        img.shape
    )



    # ----------------------
    # model
    # ----------------------

    print(
        "[LOAD POSE MODEL]"
    )


    pose=PoseRKNN(

        MODEL_PATH

    )


    print(
        "[MODEL READY]"
    )



    # ----------------------
    # inference
    # ----------------------

    print(
        "[POSE INFERENCE]"
    )


    start=time.time()



    # 注意：
    # 使用工程接口
    #
    # detect() 返回 33 个 Landmark 的列表（归一化坐标 0~1）
    # —— v5-pose 时代此函数名是 inference()，v8-pose 集成后统一为 detect()（2026-09-28 修复 P0-4）
    kpts=pose.detect(
        img
    )


    cost=(time.time()-start)*1000



    print(
        "TIME:",
        "%.2f ms"%cost
    )



    print(
        "TYPE:",
        type(kpts)
    )



    print(
        "SHAPE:",
        np.array(kpts).shape
    )



    print(
        "KEYPOINTS:"
    )


    print(
        kpts
    )



    # ----------------------
    # 检查格式
    # ----------------------


    # detect() 返回 Landmark 对象列表，需解包为 (N,3) 数组
    kpts=np.array(
        [
            [point.x, point.y, point.visibility]
            for point in kpts
        ],
        dtype=np.float32
    )



    # v8-pose 输出经 convert17to33 映射为 MediaPipe 兼容的 33 点格式
    if kpts.shape != (33,3):

        print(
            "[ERROR] keypoint format wrong"
        )

        print(
            "expect:",
            "(33,3)"
        )

        print(
            "actual:",
            kpts.shape
        )

        return



    # ----------------------
    # draw
    # ----------------------

    result=draw_pose(

        img.copy(),

        kpts

    )



    os.makedirs(
        "output",
        exist_ok=True
    )


    cv2.imwrite(

        SAVE_PATH,

        result

    )



    print(
        "SAVE:",
        SAVE_PATH
    )




    pose.release()



if __name__=="__main__":


    main()
