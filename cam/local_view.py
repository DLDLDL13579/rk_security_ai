#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板端本地实时画面显示（显示在板端屏幕上的 OpenCV 窗口）

与 mjpeg_server.py 的区别：
    mjpeg_server.py → 通过网络给远端浏览器看（Mac/手机）
    本脚本         → 在板端本机屏幕上开窗口看（HDMI 显示器）

两者可同时运行：前者拉子码流，后者拉主码流。若需省带宽，
可让本脚本也从子码流取流（CAM_CHANNEL=sub）。

窗口特性：
    · 叠加显示：帧率、分辨率、时间戳（本机时钟）
    · ESC 或 q 退出（若键盘接在板端）
    · --fullscreen 全屏
    · 断流自动重连（指数退避）

用法（板端本机，需图形环境）：
    DISPLAY=:0 python3 local_view.py --fullscreen
    DISPLAY=:0 CAM_CHANNEL=sub python3 local_view.py     # 用子码流省带宽
"""

import argparse
import os
import sys
import time

import cv2

DEFAULT_MAIN = "rtsp://admin:GKFD13258@192.168.1.64:554/Streaming/Channels/101"
DEFAULT_SUB = "rtsp://admin:GKFD13258@192.168.1.64:554/Streaming/Channels/102"

WINDOW_TITLE = os.environ.get("CAM_WINDOW_TITLE", "RK3588 Camera - Live")


def build_source(channel):
    return DEFAULT_MAIN if channel in ("main", "101") else DEFAULT_SUB


def main():
    ap = argparse.ArgumentParser(description="板端本地实时画面")
    ap.add_argument("--source", default=os.environ.get("CAM_RTSP", ""), help="RTSP 地址")
    ap.add_argument("--channel", default=os.environ.get("CAM_CHANNEL", "main"),
                    choices=["main", "sub", "101", "102"], help="取流通道")
    ap.add_argument("--width", type=int, default=int(os.environ.get("CAM_WIDTH", "0")),
                    help="显示宽度，0=原始")
    ap.add_argument("--fullscreen", action="store_true", help="全屏显示")
    args = ap.parse_args()

    source = args.source or build_source(args.channel)

    print("=" * 58)
    print(" RK3588 本地实时画面")
    print("=" * 58)
    print(f" 取流: {source}")
    print(f" DISPLAY={os.environ.get('DISPLAY', '(未设置)')}")
    print(" 退出: 按 q 或 ESC（需板端接键盘）")
    print("=" * 58)

    window_created = False
    cap = None
    backoff = 1.0
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    try:
        while True:
            if cap is None:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    print(f"[LOCAL] 连接失败，{backoff:.0f}s 后重试 ...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)
                    continue
                print("[LOCAL] 已连接")
                backoff = 1.0

            ok, frame = cap.read()
            if not ok or frame is None:
                print("[LOCAL] 读取失败，重连 ...")
                cap.release()
                cap = None
                time.sleep(1.0)
                continue

            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now

            if args.width > 0 and frame.shape[1] != args.width:
                h = int(round(frame.shape[0] * args.width / float(frame.shape[1])))
                frame = cv2.resize(frame, (args.width, h))

            # 叠加信息（本机时钟，便于与相机时间戳对比）
            from datetime import datetime

            info = f"{fps:.1f} fps | {frame.shape[1]}x{frame.shape[0]} | {datetime.now().strftime('%H:%M:%S')}"
            cv2.putText(frame, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, info, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 120), 2, cv2.LINE_AA)

            if not window_created:
                cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                if args.fullscreen:
                    cv2.setWindowProperty(WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.resizeWindow(WINDOW_TITLE, frame.shape[1], frame.shape[0])
                window_created = True

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):   # ESC / q
                print("[LOCAL] 用户退出")
                break

    except KeyboardInterrupt:
        print("\n[LOCAL] 中断退出")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[LOCAL] 已退出")

    return 0


if __name__ == "__main__":
    sys.exit(main())
