# =====================================================
# engine/rtsp_thread.py
#
# Video / RTSP Capture Thread
#
# Support:
#   rtsp://
#   mp4
#
# Output:
#   SharedData.frame
#
# =====================================================


import os
import cv2
import time
import threading
import numpy as np



class RTSPThread(threading.Thread):


    def __init__(self, shared, source):

        super().__init__()

        self.shared = shared

        self.source = source

        self.cap = None

        self.running = True
        self.frame_width = int(os.environ.get("RK_FRAME_WIDTH", "640"))
        self.frame_height = int(os.environ.get("RK_FRAME_HEIGHT", "360"))
        # 坏帧保护：低方差帧（纯黑/纯白/花屏）跳过，避免垃圾检测框
        self.frame_check = os.environ.get(
            "RK_FRAME_CHECK", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        self.frame_check_min_std = float(
            os.environ.get("RK_FRAME_CHECK_MIN_STD", "2.0")
        )


        print("[RTSP] init")




    # =====================================================
    # open
    # =====================================================

    def open_capture(self):


        print(
            "[RTSP] opening:",
            self.source
        )


        self.cap = cv2.VideoCapture(
            self.source
        )


        # 视频优化
        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )


        if not self.cap.isOpened():

            raise RuntimeError(
                "Cannot open source"
            )


        print("[RTSP] opened")



    def _letterbox(self, frame, target_size):

        """保持宽高比缩放到 target_size (w, h)，不足处补黑边，避免人物拉伸变形。"""

        tw, th = target_size
        h, w = frame.shape[:2]
        scale = min(tw / float(w), th / float(h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        x0 = (tw - nw) // 2
        y0 = (th - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas



    # =====================================================
    # run
    # =====================================================

    def run(self):


        print(
            "[RTSP Thread] started"
        )


        try:


            self.open_capture()


            while self.shared.running:



                ret, frame = self.cap.read()



                if not ret:


                    print(
                        "[RTSP] read failed"
                    )


                    # mp4 循环

                    if isinstance(
                        self.source,
                        str
                    ) and self.source.endswith(
                        (
                            ".mp4",
                            ".avi",
                            ".mkv"
                        )
                    ):


                        self.cap.release()


                        time.sleep(0.2)


                        self.open_capture()


                        continue



                    time.sleep(0.05)


                    continue




                # =========================
                # 最新帧覆盖
                # =========================


                if (
                    self.frame_width > 0
                    and self.frame_height > 0
                    and (
                        frame.shape[1] != self.frame_width
                        or frame.shape[0] != self.frame_height
                    )
                ):
                    frame = self._letterbox(
                        frame,
                        (self.frame_width, self.frame_height),
                    )

                if self.frame_check:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if float(gray.std()) < self.frame_check_min_std:
                        time.sleep(0.01)
                        continue

                self.shared.set_frame(
                    frame
                )



                # 防止CPU占满

                time.sleep(
                    0.005
                )



        except Exception as e:


            print(
                "[RTSP ERROR]",
                e
            )


            self.shared.running=False




        finally:


            if self.cap:


                self.cap.release()



            print(
                "[RTSP] stopped"
            )





    # =====================================================
    # stop
    # =====================================================

    def stop(self):


        print(
            "[RTSP] stopping"
        )


        self.shared.running=False


        if self.cap:


            self.cap.release()
