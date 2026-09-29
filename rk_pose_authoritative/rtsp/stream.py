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


            # === RTSP_STARTUP_RETRY_PATCH (2026-09-29) ===
            # 实时流首次连接失败时重试（相机晚就绪不该导致整条流水线退出）；
            # 本地视频文件保持原行为：打不开即报错退出。
            _startup_attempts = 0
            while True:
                try:
                    self.open_capture()
                    break
                except Exception as _open_exc:
                    if self._is_file_source() or not self.shared.running:
                        raise
                    _startup_attempts += 1
                    _wait = min(2.0 ** min(_startup_attempts, 5), 30.0)
                    print(
                        "[RTSP] 首次连接失败（第 %d 次）：%s —— %.1fs 后重试"
                        % (_startup_attempts, _open_exc, _wait)
                    )
                    _slept = 0.0
                    while _slept < _wait:
                        if not self.shared.running:
                            raise
                        time.sleep(min(0.5, _wait - _slept))
                        _slept += 0.5


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



                    # === RTSP_RECONNECT_PATCH (2026-09-29) ===
                    # 原实现此处仅 sleep 后 continue，RTSP 流断开后线程永久空转、
                    # 永不重连（实测 eth2 故障后空转近 2 小时）。
                    # 现改为按退避策略重建连接，成功读帧后退避重置。
                    if self._rtsp_reconnect():
                        continue
                    # 重试耗尽或收到停止信号
                    if not self.shared.running:
                        break
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

                if hasattr(self, "_rtsp_backoff"):
                    self._rtsp_backoff = 1.0

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






    # === RTSP_RECONNECT_PATCH (2026-09-29) ===
    def _is_file_source(self):
        """判断输入是本地视频文件而非实时流"""
        return isinstance(self.source, str) and self.source.lower().endswith(
            (".mp4", ".avi", ".mkv", ".mov", ".flv")
        )

    def _rtsp_reconnect(self):
        """
        重建 RTSP 连接（指数退避 1s→2s→4s…上限 30s）。

        返回 True 表示已重连成功（调用方应 continue 继续读帧）；
        返回 False 表示停止或重试期间收到退出信号。
        """
        if self._is_file_source():
            return False

        if not hasattr(self, "_rtsp_backoff"):
            self._rtsp_backoff = 1.0

        max_backoff = float(os.environ.get("RK_RTSP_MAX_BACKOFF", "30"))
        delay = self._rtsp_backoff

        print(f"[RTSP] 流中断，{delay:.1f}s 后重连 ...")

        # 分片睡眠，保证能及时响应停止信号
        waited = 0.0
        while waited < delay:
            if not self.shared.running:
                return False
            time.sleep(min(0.5, delay - waited))
            waited += 0.5

        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            time.sleep(0.5)
            self.open_capture()
            self._rtsp_backoff = 1.0
            print("[RTSP] 重连成功")
            return True
        except Exception as exc:
            print(f"[RTSP] 重连失败: {exc}")
            self._rtsp_backoff = min(delay * 2.0, max_backoff)
            return True  # 返回 True 让循环继续（下一轮再按新退避重试）

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
