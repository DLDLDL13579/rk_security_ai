import os
# ==========================================================
# record_rtsp.py
#
# RK3588 Hikvision RTSP Recorder
#
# RTSP Camera
#       |
#       |
# OpenCV
#       |
#       |
# MP4 Video
#
# ==========================================================


import cv2
import time
import os


# ==========================================================
# RTSP配置
# ==========================================================

RTSP_URL = os.environ.get("RK_RTSP_URL", "")
# 默认示例（请用环境变量 RK_RTSP_URL 覆盖）:
#   rtsp://<user>:<password>@<ip>:554/Streaming/Channels/101


# ==========================================================
# 保存配置
# ==========================================================

SAVE_DIR = "videos_camera"


# 修改这里：
VIDEO_NAME = "bend.mp4"


SAVE_PATH = os.path.join(
    SAVE_DIR,
    VIDEO_NAME
)



# ==========================================================
# 创建目录
# ==========================================================

if not os.path.exists(SAVE_DIR):

    os.makedirs(SAVE_DIR)



print("="*60)
print(" RK3588 RTSP VIDEO RECORDER ")
print("="*60)


print("Save:",
      SAVE_PATH)



# ==========================================================
# 打开RTSP
# ==========================================================

print("[RTSP] Connecting...")


cap = cv2.VideoCapture(
    RTSP_URL,
    cv2.CAP_FFMPEG
)


# 设置缓存
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)



if not cap.isOpened():

    raise RuntimeError(
        "RTSP open failed"
    )


print("[RTSP] Connected")



# ==========================================================
# 获取视频参数
# ==========================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)


width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)



# 有些海康返回0

if fps <= 0:

    fps = 25



if width <=0 or height<=0:

    width=1920
    height=1080



print(
    "Resolution:",
    width,
    "x",
    height
)


print(
    "FPS:",
    fps
)



# ==========================================================
# 视频编码
# ==========================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(

    SAVE_PATH,

    fourcc,

    fps,

    (
        width,
        height
    )

)


if not writer.isOpened():

    raise RuntimeError(
        "Video writer failed"
    )


print("[Recorder] Start")

print("ESC stop recording")



# ==========================================================
# FPS统计
# ==========================================================


frame_count=0

start=time.time()

show_fps=0



# ==========================================================
# 主循环
# ==========================================================

while True:


    ret,frame = cap.read()



    if not ret:

        print(
            "[WARN] frame lost"
        )

        time.sleep(
            0.05
        )

        continue



    frame_count +=1



    # 写入视频

    writer.write(
        frame
    )



    # FPS

    if time.time()-start>=1:


        show_fps = (
            frame_count /
            (time.time()-start)
        )


        frame_count=0

        start=time.time()



    # 显示

    cv2.putText(

        frame,

        f"REC FPS:{show_fps:.1f}",

        (30,50),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0,255,0),

        2

    )


    cv2.imshow(
        "RTSP Recorder",
        frame
    )



    key=cv2.waitKey(1)



    if key==27:

        break




# ==========================================================
# 释放
# ==========================================================

print("\nStopping...")


cap.release()

writer.release()

cv2.destroyAllWindows()



print(
    "Saved:",
    SAVE_PATH
)


print(
    "Done"
)
