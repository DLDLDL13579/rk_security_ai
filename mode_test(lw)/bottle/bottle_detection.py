#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
瓶子检测 - ONNX Runtime 版（带详细调试输出）
支持 CPU/GPU 推理，自动适配输入名称，打印原始检测结果。
"""

import os
import sys
import cv2
import numpy as np
import time
import re
import onnxruntime as ort

# ========== 配置参数（请根据实际情况修改） ==========
ONNX_PATH = "/home/nvidia/test/PaddleDetection/mode_test(lw)/bottle/picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet.onnx"
RTSP_URL = os.environ.get("RK_RTSP_URL", "")
# 调试阶段建议先设为 0.01，看有无低分框
CONF_THRESHOLD = 0.01
BOTTLE_CLASS_ID = 44  # 标准 COCO bottle 类别 ID，若实际不同请修改
INPUT_SIZE = 416  # 模型输入尺寸（必须与 ONNX 一致）
DISPLAY_SIZE = (640, 480)
FRAME_SKIP = 2
SHOW_FPS = True


# ===================================================

class ONNXDetector:
    def __init__(self, onnx_path):
        # 尝试 GPU，若失败则 CPU
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            print("[INFO] 使用 GPU 推理")
        except:
            self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            print("[INFO] 使用 CPU 推理")

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        print("[INFO] 模型输入:", [(inp.name, inp.shape, inp.type) for inp in inputs])
        print("[INFO] 模型输出:", [(out.name, out.shape, out.type) for out in outputs])

        # 获取输入名称
        self.input_name_image = None
        self.input_name_scale = None
        for inp in inputs:
            name = inp.name
            if 'image' in name.lower():
                self.input_name_image = name
            elif 'scale' in name.lower():
                self.input_name_scale = name
        if self.input_name_image is None and len(inputs) > 0:
            self.input_name_image = inputs[0].name
            print(f"[WARN] 未找到 'image' 输入，使用 {self.input_name_image}")
        if self.input_name_scale is None:
            print("[INFO] 模型不需要 scale_factor 输入")

        # 预处理参数（标准 ImageNet）
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.input_size = INPUT_SIZE

    def preprocess(self, img):
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = img_rgb.astype(np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, :, :, :]
        return img_array.astype(np.float32)

    def predict(self, img):
        input_tensor = self.preprocess(img)
        feed_dict = {self.input_name_image: input_tensor}
        if self.input_name_scale is not None:
            feed_dict[self.input_name_scale] = np.array([[1.0, 1.0]], dtype=np.float32)
        outputs = self.session.run(None, feed_dict)
        out = outputs[0]
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]  # (num_dets, 6)

        # ========== 调试：打印原始输出信息 ==========
        print(f"[DEBUG] 输出 shape: {out.shape}, dtype: {out.dtype}")
        if out.size == 0:
            print("[DEBUG] 无检测结果")
            return [], []

        # 打印前5个检测框原始数据
        print("[DEBUG] 前5个检测框原始数据 (每行: x1, y1, x2, y2, score, class_id):")
        for i in range(min(5, len(out))):
            det = out[i]
            print(f"  [{i}] {det}")

        # 打印所有检测框的分数和类别ID（不限阈值）
        print("[DEBUG] 所有检测框 (score, class_id):")
        for det in out:
            if len(det) >= 6:
                print(f"      score={det[4]:.4f}, class={int(det[5])}")
        # ============================================

        # 后处理：坐标缩放和过滤
        scale_x = img.shape[1] / self.input_size
        scale_y = img.shape[0] / self.input_size
        boxes, scores = [], []
        for det in out:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, score, cls_id = det[:6]
            if score >= CONF_THRESHOLD and int(cls_id) == BOTTLE_CLASS_ID:
                boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                scores.append(score)

        print(f"[INFO] 最终检测到 {len(boxes)} 个瓶子")
        if len(boxes) > 0:
            print(f"      最高置信度: {max(scores):.3f}")
        return boxes, scores


def draw_boxes(image, boxes, scores):
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"bottle: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image


def open_rtsp_stream(rtsp_url):
    pattern = r'rtsp://([^:]+):([^@]+)@(.+)$'
    match = re.match(pattern, rtsp_url)
    if not match:
        print("[ERROR] RTSP URL格式错误")
        return None
    user, password, host_path = match.groups()
    base_url = f"rtsp://{host_path}"
    decoders = [
        ("H265", "rtph265depay ! nvv4l2decoder"),
        ("H264", "rtph264depay ! nvv4l2decoder"),
        ("H265", "rtph265depay ! avdec_h265"),
        ("H264", "rtph264depay ! avdec_h264")
    ]
    for codec_name, decoder in decoders:
        gst_pipeline = (
            f"rtspsrc location={base_url} user-id={user} user-pw={password} "
            "latency=0 drop-on-latency=true tcp-timeout=0 ! "
            f"{decoder} ! videoconvert ! video/x-raw,format=BGR ! appsink"
        )
        print(f"[INFO] 尝试GStreamer ({codec_name})...")
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] 成功打开RTSP流 ({codec_name})")
                return cap
            cap.release()
    print("[WARN] GStreamer失败，使用OpenCV FFMPEG")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("[INFO] OpenCV FFMPEG打开成功")
        return cap
    print("[ERROR] 无法打开RTSP流")
    return None


def main():
    print("[INFO] 加载ONNX模型...")
    if not os.path.exists(ONNX_PATH):
        print(f"[ERROR] ONNX模型文件不存在: {ONNX_PATH}")
        sys.exit(1)
    detector = ONNXDetector(ONNX_PATH)
    print("[INFO] 模型加载完成")

    cap = open_rtsp_stream(RTSP_URL)
    if cap is None:
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    last_boxes, last_scores = [], []
    fps = 0
    fps_start = time.time()
    fps_counter = 0

    print("[INFO] 开始检测，按 'q' 退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取失败，尝试重连...")
            cap.release()
            cap = open_rtsp_stream(RTSP_URL)
            if cap is None:
                break
            continue

        frame_count += 1
        fps_counter += 1

        if frame_count % FRAME_SKIP == 0:
            boxes, scores = detector.predict(frame)
            last_boxes, last_scores = boxes, scores
        else:
            boxes, scores = last_boxes, last_scores

        vis = draw_boxes(frame.copy(), boxes, scores)

        if SHOW_FPS:
            if fps_counter >= 30:
                now = time.time()
                fps = fps_counter / (now - fps_start)
                fps_start = now
                fps_counter = 0
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if DISPLAY_SIZE:
            vis = cv2.resize(vis, DISPLAY_SIZE)

        cv2.imshow("Bottle Detection (ONNX)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 退出")


if __name__ == "__main__":
    main()