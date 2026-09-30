import os
import cv2
import os

# === DISPLAY_FULLSCREEN_PATCH (2026-09-30) ===
WINDOW_NAME = "RK3588 Smart Security"
import time


# ============================================================
# 骨架连线定义（2026-09-28 新增：支持画人体骨架）
# ============================================================

# MediaPipe 33 点骨架（本项目特征提取用的体系）
SKELETON_33 = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # 双臂+肩
    (11, 23), (12, 24), (23, 24),                        # 躯干
    (23, 25), (25, 27), (24, 26), (26, 28),              # 双腿
    (27, 31), (28, 32), (15, 17), (16, 18),              # 手脚末端
    (0, 11), (0, 12),                                    # 头->肩
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),             # 面部
]

# COCO 17 点骨架（v8-pose 原生输出，未经映射时用）
SKELETON_17 = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# 骨架配色（BGR）
SKELETON_COLOR = (255, 128, 0)      # 蓝色线条
KEYPOINT_COLOR = (0, 0, 255)        # 红色关节点
KP_VIS_THRESHOLD = 0.3              # 关键点可见性阈值


class DisplayThread:
    def __init__(self, shared, width=None, height=None):
        self.shared = shared
        # === DISPLAY_FULLSCREEN_PATCH (2026-09-30) ===
        # 板端 HDMI 显示：默认按屏幕分辨率铺满（1920x1080），可用环境变量覆盖。
        # 原默认 960x540 会在 1080p 屏上留出大片桌面。
        _dw = int(os.environ.get("RK_DISPLAY_WIDTH", "1920"))
        _dh = int(os.environ.get("RK_DISPLAY_HEIGHT", "1080"))
        self.width = _dw if width is None else width
        self.height = _dh if height is None else height
        self.fullscreen = os.environ.get(
            "RK_DISPLAY_FULLSCREEN", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        self._window_ready = False

        self.last_time = time.time()
        self.fps = 0.0
        self.track_motion = {}
        self.predict_lag_frames = max(
            0.0,
            float(os.environ.get("RK_DISPLAY_PREDICT_LAG", "2.6")),
        )
        self.predict_alpha = min(
            0.95,
            max(0.0, float(os.environ.get("RK_DISPLAY_PREDICT_ALPHA", "0.48"))),
        )
        self.headless = os.environ.get(
            "RK_HEADLESS", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_path = os.environ.get(
            "RK_OUTPUT_VIDEO",
            os.path.join(project_root, "output", "behavior_result.mp4"),
        )
        self.video_writer = None

    def _ensure_writer(self, frame_shape):
        if self.video_writer is not None:
            return
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (self.width, self.height),
        )
        print("[Display] headless recording ->", self.output_path)

    def update_fps(self):
        now = time.time()
        dt = now - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = now

    def clip_bbox(self, bbox, width, height):
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width - 1))
        y2 = max(0, min(int(round(y2)), height - 1))
        return [x1, y1, x2, y2]

    def scale_bbox(self, bbox, src_width, src_height):
        if src_width <= 0 or src_height <= 0:
            return bbox

        scale_x = self.width / float(src_width)
        scale_y = self.height / float(src_height)
        x1, y1, x2, y2 = bbox
        return [
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y,
        ]

    def predict_bbox(self, pid, bbox, result_frame_id, current_frame_id):
        x1, y1, x2, y2 = map(float, bbox)
        lag_frames = max(0, current_frame_id - result_frame_id)
        mem = self.track_motion.get(pid)

        if mem is None:
            self.track_motion[pid] = {
                "bbox": [x1, y1, x2, y2],
                "frame_id": result_frame_id,
                "vx": 0.0,
                "vy": 0.0,
            }
            return [x1, y1, x2, y2]

        last_bbox = mem["bbox"]
        last_frame_id = mem["frame_id"]
        dt = max(1, result_frame_id - last_frame_id)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        last_cx = (last_bbox[0] + last_bbox[2]) * 0.5
        last_cy = (last_bbox[1] + last_bbox[3]) * 0.5

        vx = (cx - last_cx) / dt
        vy = (cy - last_cy) / dt

        mem["bbox"] = [x1, y1, x2, y2]
        mem["frame_id"] = result_frame_id
        mem["vx"] = (1.0 - self.predict_alpha) * mem["vx"] + self.predict_alpha * vx
        mem["vy"] = (1.0 - self.predict_alpha) * mem["vy"] + self.predict_alpha * vy

        lag = min(self.predict_lag_frames, float(lag_frames) + 0.6)
        dx = mem["vx"] * lag
        dy = mem["vy"] * lag

        # === PREDICT_CLAMP_PATCH_20260930 ===
        # 外推位移上限：不超过框自身尺寸的一半，且不超过画面尺寸的 1/8。
        # 原实现无上限，人快速走动时框被推出屏幕外（用户实测反馈）。
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        max_dx = min(box_w * 0.5, self.width / 8.0)
        max_dy = min(box_h * 0.5, self.height / 8.0)
        if abs(dx) > max_dx:
            dx = max_dx if dx > 0 else -max_dx
        if abs(dy) > max_dy:
            dy = max_dy if dy > 0 else -max_dy

        return [
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy,
        ]

    # ========================================================
    # 画人体骨架（2026-09-28 新增）
    # ========================================================

    def draw_skeleton(self, img, kpts, frame_mode="full_frame", crop_box=None,
                      src_shape=None, dst_shape=None):
        """在 img 上画骨架

        kpts:        [{"x","y","v"}, ...] 33 个关键点
        frame_mode:  "full_frame" = 坐标为整帧归一化 0~1
                     "crop"       = 坐标为裁剪框内归一化，需换算
        crop_box:    (x1,y1,x2,y2) 裁剪框（crop 模式需要），源图坐标系
        src_shape:   源图 (h, w)
        dst_shape:   目标显示图 (h, w)
        """
        if not kpts:
            return

        h_dst, w_dst = dst_shape if dst_shape else img.shape[:2]

        # 选择骨架连表：33 点用 SKELETON_33，17 点用 SKELETON_17
        if len(kpts) >= 33:
            links = SKELETON_33
        elif len(kpts) >= 17:
            links = SKELETON_17
        else:
            return

        # 计算归一化 -> 显示像素的映射
        pts = []
        if frame_mode == "full_frame" or crop_box is None:
            # 整帧归一化：直接乘显示尺寸
            for kp in kpts:
                pts.append((int(kp["x"] * w_dst), int(kp["y"] * h_dst), kp.get("v", 1.0)))
        else:
            # 裁剪框内归一化：先换算到源图，再缩放到显示图
            if src_shape is None:
                return
            h_src, w_src = src_shape
            cx1, cy1, cx2, cy2 = crop_box
            cw, ch = max(1, cx2 - cx1), max(1, cy2 - cy1)
            sx = w_dst / float(w_src)
            sy = h_dst / float(h_src)
            for kp in kpts:
                px = (cx1 + kp["x"] * cw) * sx
                py = (cy1 + kp["y"] * ch) * sy
                pts.append((int(px), int(py), kp.get("v", 1.0)))

        # 画连线
        for a, b in links:
            if a >= len(pts) or b >= len(pts):
                continue
            pa, pb = pts[a], pts[b]
            if pa[2] >= KP_VIS_THRESHOLD and pb[2] >= KP_VIS_THRESHOLD:
                cv2.line(img, (pa[0], pa[1]), (pb[0], pb[1]),
                         SKELETON_COLOR, 2, cv2.LINE_AA)

        # 画关节点
        for px, py, v in pts:
            if v >= KP_VIS_THRESHOLD:
                cv2.circle(img, (px, py), 3, KEYPOINT_COLOR, -1, cv2.LINE_AA)

    def draw_person(self, img, pid, det, behavior, result_frame_id, current_frame_id, source_shape):
        bbox = det.get("bbox")
        if bbox is None:
            return

        predicted_bbox = self.predict_bbox(
            pid,
            bbox,
            result_frame_id,
            current_frame_id,
        )
        scaled_bbox = self.scale_bbox(
            predicted_bbox,
            source_shape[1],
            source_shape[0],
        )
        x1, y1, x2, y2 = self.clip_bbox(scaled_bbox, self.width, self.height)

        if x2 <= x1 or y2 <= y1:
            return

        score = float(det.get("score", 0.0))
        name = det.get("name", "person")
        action = ""
        action_score = 0.0
        status = ""

        if behavior is not None:
            action = behavior.get("action", "")
            action_score = float(behavior.get("confidence", 0.0))
            status = behavior.get("state", "")

        color = (0, 255, 0)
        if action == "fall_down":
            color = (0, 0, 255)
        elif status in ("ambiguous", "contaminated", "partial", "low_quality"):
            color = (0, 165, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{pid} {name} {score:.2f}"
        if action:
            label += f" | {action} {action_score:.2f}"
        if status not in ("", "valid", "warming"):
            label += f" [{status}]"

        font_scale = 0.85
        thickness = 2
        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness,
        )

        # === LABEL_INSIDE_PATCH_20260930 ===
        # 原实现把标签画在框上方（y1 - th - 12）。人位于画面顶部时 y1 很小，
        # 标签被 clamp 到 0 后遭画面顶部裁掉 —— 用户看不到行为识别结果。
        # 改为画在「框内顶端」：框内必然可见。框过矮时退化为框内底部。
        if (y2 - y1) >= (th + 14):
            label_top = y1 + 4            # 框内顶端
        else:
            label_top = max(y1, y2 - th - 6)   # 框太矮 → 框内底部
        label_bottom = min(self.height - 1, label_top + th + 10)
        # 修正：标签起点也要 clamp，否则人在画面右侧时文字被右边缘裁掉
        # （2026-09-28 修复：原来只 clamp 右边界，导致可用宽度不足）
        box_w = tw + 10
        label_left = max(0, min(x1, self.width - 1 - box_w))
        label_right = min(self.width - 1, label_left + box_w)

        cv2.rectangle(
            img,
            (label_left, label_top),
            (label_right, label_bottom),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            label,
            (label_left + 5, label_bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 255),
            thickness,
        )

    def run(self):
        print("[Display] Thread Started")

        while self.shared.running:
            detection_packet = self.shared.get_detection_packet()
            frame, current_frame_id = self.shared.get_frame_packet(copy_frame=True)

            if frame is None:
                frame = detection_packet.get("frame")
                current_frame_id = detection_packet.get("frame_id", -1)

            result_frame_id = detection_packet.get("frame_id", -1)
            frame = frame if frame is not None else detection_packet.get("frame")

            if frame is None:
                time.sleep(0.005)
                continue

            source_shape = frame.shape[:2]
            show = cv2.resize(
                frame,
                (self.width, self.height),
            )

            detections = detection_packet.get("detections", {})
            behaviors = self.shared.get_behaviors()

            if isinstance(detections, dict):
                for pid, det in detections.items():
                    behavior = behaviors.get(pid) if isinstance(behaviors, dict) else None

                    # 先画骨架（在框下层，避免遮挡），2026-09-28 新增
                    if behavior is not None and behavior.get("keypoints"):
                        self.draw_skeleton(
                            show,
                            behavior.get("keypoints"),
                            frame_mode=behavior.get("keypoints_frame", "full_frame"),
                            crop_box=behavior.get("crop_box"),
                            src_shape=source_shape,
                            dst_shape=show.shape[:2],
                        )

                    self.draw_person(
                        show,
                        pid,
                        det,
                        behavior,
                        result_frame_id,
                        current_frame_id,
                        source_shape,
                    )

            self.update_fps()
            self.shared.set_fps(self.fps)

            cv2.putText(
                show,
                f"FPS:{self.fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
            )

            if self.headless:
                self._ensure_writer(show.shape)
                if self.video_writer is not None:
                    self.video_writer.write(show)
                time.sleep(0.005)
            else:
                # === DISPLAY_FULLSCREEN_PATCH (2026-09-30) ===
                # 首次显示前创建全屏窗口，让画面铺满 HDMI 屏
                if not self._window_ready:
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    if self.fullscreen:
                        cv2.setWindowProperty(
                            WINDOW_NAME,
                            cv2.WND_PROP_FULLSCREEN,
                            cv2.WINDOW_FULLSCREEN,
                        )
                    self._window_ready = True

                cv2.imshow(WINDOW_NAME, show)
                key = cv2.waitKey(1)
                if key == 27:
                    self.shared.running = False
                    break

        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("[Display] stopped")
