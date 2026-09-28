# =====================================================
# engine/rtsp_thread.py
#
# Video / RTSP Capture Thread
#
# Support:
#   rtsp://
#   mp4 / avi / mkv / mov
#
# Output:
#   SharedData.frame
#
# =====================================================


import threading
import time

import cv2


VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".flv")


class RTSPThread(threading.Thread):
    def __init__(self, shared, source, loop_video=False):
        super().__init__()
        self.shared = shared
        self.source = source
        self.loop_video = bool(loop_video)
        self.cap = None
        self.running = True
        self.source_is_file = False
        self.source_fps = 0.0
        self.frame_interval = 0.0
        print("[RTSP] init")

    def _detect_file_source(self):
        return isinstance(self.source, str) and self.source.lower().endswith(VIDEO_SUFFIXES)

    def open_capture(self):
        print("[RTSP] opening:", self.source)
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open source")

        self.source_is_file = self._detect_file_source()
        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1.0 or fps > 240.0:
            fps = 25.0 if self.source_is_file else 0.0
        self.source_fps = fps
        self.frame_interval = 1.0 / fps if fps > 0 else 0.0

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.shared.set_source_info(
            fps=self.source_fps,
            is_file=self.source_is_file,
            size=(width, height),
        )

        print("[RTSP] opened")
        if self.source_is_file:
            print(f"[RTSP] file fps={self.source_fps:.2f} loop={self.loop_video}")

    def run(self):
        print("[RTSP Thread] started")
        try:
            self.open_capture()
            next_tick = time.time()

            while self.shared.running:
                ret, frame = self.cap.read()

                if not ret:
                    print("[RTSP] read failed")

                    if self.source_is_file and self.loop_video:
                        self.cap.release()
                        time.sleep(0.2)
                        self.open_capture()
                        next_tick = time.time()
                        continue

                    if self.source_is_file:
                        print("[RTSP] file end")
                        self.shared.running = False
                        break

                    time.sleep(0.05)
                    continue

                self.shared.set_frame(frame)

                if self.source_is_file and self.frame_interval > 0:
                    next_tick += self.frame_interval
                    sleep_time = next_tick - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        next_tick = time.time()
                else:
                    time.sleep(0.005)

        except Exception as e:
            print("[RTSP ERROR]", e)
            self.shared.running = False

        finally:
            if self.cap:
                self.cap.release()
            print("[RTSP] stopped")

    def stop(self):
        print("[RTSP] stopping")
        self.shared.running = False
        if self.cap:
            self.cap.release()
