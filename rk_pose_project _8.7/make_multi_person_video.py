# -*- coding: utf-8 -*-
"""
============================================================
make_multi_person_video.py

把两路单人视频合成一路"两人同框"的测试视频（左右并排）。
这样无需真实多人素材，也能立刻测试多人的跟踪与行为识别，
并且可以知道画面里每个人的真实动作（来自原始视频目录名）。

用法:
    python make_multi_person_video.py \
        --left videos/standing/stand_001.mp4 \
        --right videos/walking/walk_001.mp4 \
        --out videos_multi/m1_stand_walk.mp4

可选参数:
    --flip-right   右侧视频左右翻转（让两人朝向不同，更像真实场景）
    --height 360   输出高度（宽度自动 2x）
    --fps 20       输出帧率
============================================================
"""

import argparse
import os

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="merge two videos side by side")
    parser.add_argument("--left", required=True, help="left video path")
    parser.add_argument("--right", required=True, help="right video path")
    parser.add_argument("--out", required=True, help="output video path")
    parser.add_argument("--flip-right", action="store_true",
                        help="mirror right video horizontally")
    parser.add_argument("--height", type=int, default=360,
                        help="output height (default 360)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="output fps (default 20)")
    args = parser.parse_args()

    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)
    if not cap_l.isOpened():
        raise FileNotFoundError(f"cannot open left video: {args.left}")
    if not cap_r.isOpened():
        raise FileNotFoundError(f"cannot open right video: {args.right}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    half_w = args.height * 2  # 4:3 单侧宽高比
    writer = cv2.VideoWriter(
        args.out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (half_w * 2, args.height),
    )

    frame_id = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        # 保持宽高比缩放 + 黑边（letterbox），避免人物拉伸变形
        frame_l = _letterbox(frame_l, (half_w, args.height))
        frame_r = _letterbox(frame_r, (half_w, args.height))
        if args.flip_right:
            frame_r = cv2.flip(frame_r, 1)

        canvas = np.hstack([frame_l, frame_r])
        writer.write(canvas)
        frame_id += 1

        if frame_id % 30 == 0:
            print("[MERGE]", frame_id, "frames")

    cap_l.release()
    cap_r.release()
    writer.release()
    print("[DONE] saved:", args.out, "frames:", frame_id)


def _letterbox(frame, target_size):
    """缩放并居中到 target_size (w, h)，不足处补黑边，保持宽高比。"""
    tw, th = target_size
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


if __name__ == "__main__":
    main()
