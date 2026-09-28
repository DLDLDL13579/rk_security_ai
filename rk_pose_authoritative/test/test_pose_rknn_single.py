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

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.rknn")

IMAGE_PATH = os.path.join(PROJECT_ROOT, "test", "person.jpg")

SAVE_PATH = os.path.join(PROJECT_ROOT, "output", "pose_test.jpg")



# ==========================
# COCO17骨架
# ==========================

SKELETON = [

    (0,1),
    (0,2),

    (1,3),
    (2,4),

    (5,6),

    (5,7),
    (7,9),

    (6,8),
    (8,10),

    (5,11),
    (6,12),

    (11,12),

    (11,13),
    (13,15),

    (12,14),
    (14,16)

]



def draw_pose(
        img,
        kpts
):


    for x,y,score in kpts:


        if score > 0.2:


            cv2.circle(

                img,

                (
                    int(x),
                    int(y)
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
                    int(kpts[a][0]),
                    int(kpts[a][1])
                ),

                (
                    int(kpts[b][0]),
                    int(kpts[b][1])
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
    kpts=pose.inference(
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


    kpts=np.array(
        kpts,
        dtype=np.float32
    )



    if kpts.shape != (17,3):

        print(
            "[ERROR] keypoint format wrong"
        )

        print(
            "expect:",
            "(17,3)"
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
