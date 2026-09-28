# -*- coding: utf-8 -*-
"""
============================================================
engine/locomotion_engine.py

Locomotion 通道：仅负责 standing / walking。

判定依据（全部为逐帧运动学/步态证据，不依赖 TCN 序列）：
  - motion       : 检测框中心位移（按身高归一化，EMA）
  - pose_energy  : 下半身关键点相对髋部的运动能量
  - gait         : 踝关节 y 向摆动幅度 + 换向节奏（步态证据）

输出带滞回阈值 + 连续帧确认，避免 standing/walking 频繁抖动。
============================================================
"""

import os
from collections import deque

from engine.geometry import ankle_amp, cadence


class LocomotionEngine:

    def __init__(self):
        self.enter_motion = float(
            # 【2026-09-28 实测重标定：严进宽出策略】
            # v8-pose 的位移特征比 MediaPipe 时代缩小到 1/3~1/2，旧值 0.016 是
            # walk_001 的 motion 中位数（0.0049）的 3.3 倍 → walking 全被判成 standing。
            #
            # 但 standing 与 walking 的 motion 分布严重重叠（实测前 250 帧 × 5 视频）：
            #   standing: p25=0.0026 median=0.0038 p75=0.0053 p90=0.0082
            #   walking : p25=0.0042 median=0.0086 p75=0.0192 p90=0.0432
            # 单一阈值无法干净切分——低 enter 会让 standing 误判成 walking。
            #
            # 故采用「严进门 + 宽保持」：
            #   enter=0.010（≥0.010 时 standing 仅 6.9% 而 walking 有 42.4%，区分度最好）
            #   exit =0.0015（进入后靠迟滞保持，避免 motion 波动掉回 standing）
            # 50 视频抽样对比（10 视频/类）：
            #   enter/exit 0.005/0.0015 -> standing 30~40% / walking 90%
            #   enter/exit 0.010/0.0015 -> standing  50%    / walking 80% / squat 80%
            os.environ.get("RK_LOCO_ENTER_MOTION", "0.010")
        )
        self.exit_motion = float(
            # exit 是 walking 的"保持门"：旧值 0.008 高于 walking 的 motion 中位（0.0049），
            # 人走着稍有波动就掉回 standing 并被迟滞锁死。
            # 单视频实测：exit 0.003→standing（walking 票 133）；0.0015→walking（票 350）。
            os.environ.get("RK_LOCO_EXIT_MOTION", "0.0015")
        )
        self.enter_pose_energy = float(
            os.environ.get("RK_LOCO_ENTER_POSE_ENERGY", "0.008")
        )
        self.exit_pose_energy = float(
            os.environ.get("RK_LOCO_EXIT_POSE_ENERGY", "0.004")
        )
        self.gait_amp = float(
            os.environ.get("RK_LOCO_GAIT_AMP", "0.006")
        )
        self.amp_strong = float(
            os.environ.get("RK_LOCO_GAIT_AMP_STRONG", "0.025")
        )
        self.ankle_vis_threshold = float(
            os.environ.get("RK_LOCO_ANKLE_VIS", "0.40")
        )
        self.gait_cadence = max(
            2,
            int(os.environ.get("RK_LOCO_GAIT_CADENCE", "3")),
        )
        self.stand_amp_max = float(
            os.environ.get("RK_LOCO_STAND_AMP_MAX", "0.004")
        )
        self.min_pose_quality = float(
            os.environ.get("RK_LOCO_MIN_POSE_QUALITY", "0.28")
        )
        self.switch_streak = max(
            1,
            int(os.environ.get("RK_LOCO_SWITCH_STREAK", "3")),
        )
        self.warmup_samples = max(
            3,
            int(os.environ.get("RK_LOCO_WARMUP_SAMPLES", "5")),
        )
        self.walk_hint_score = float(
            os.environ.get("RK_LOCO_WALK_HINT_SCORE", "0.68")
        )
        self.gait_window = max(
            6,
            int(os.environ.get("RK_LOCO_GAIT_WINDOW", "12")),
        )
        self.states = {}

        print("[Locomotion] Ready (geometry + gait)")

    def clear_track(self, track_id):
        self.states.pop(track_id, None)

    def resolve(
        self,
        track_id,
        motion,
        pose_energy,
        pose_quality,
        keypoints=None,
        model_action="",
        model_score=0.0,
        geometry=None,
    ):
        state = self.states.get(track_id)
        if state is None:
            state = {
                "action": "standing",
                "score": 0.75,
                "candidate_action": "standing",
                "candidate_count": 0,
                "samples": 0,
                "kpts_history": deque(maxlen=self.gait_window),
            }
            self.states[track_id] = state

        state["samples"] += 1
        if keypoints is not None and len(keypoints) >= 29:
            state["kpts_history"].append(keypoints)

        points = list(state["kpts_history"])
        amp = ankle_amp(points)
        cad = cadence(points)

        if pose_quality < self.min_pose_quality:
            return {
                "action": state["action"],
                "confidence": float(state["score"]),
                "state": "held",
                "channel": "locomotion",
            }

        motion_ev = motion >= self.enter_motion
        gait_ev = amp >= self.gait_amp and cad >= self.gait_cadence
        gait_ev_strong = amp >= self.amp_strong and cad >= self.gait_cadence
        model_ev = (
            model_action == "walking"
            and model_score >= self.walk_hint_score
        )

        # 小尺寸人物裁剪下 pose_energy 噪声可达 0.05~0.3（远超静止阈值），
        # 不再作为行走触发；行走以 bbox 位移为准。
        # 步态证据必须通过踝关节可见性门：踝点不可见时，amp 是关键点失败/幻觉
        # 产生的垃圾数据（实测可达 0.17~4.9），不可作为行走依据。
        valid_pts = [p for p in points if p is not None and len(p) >= 29]
        ankle_vis = 0.0
        if valid_pts:
            ankle_vis = sum(
                min(p[27].visibility, p[28].visibility)
                for p in valid_pts
            ) / len(valid_pts)
        gait_valid = (
            ankle_vis >= self.ankle_vis_threshold
            and amp <= 1.0
        )
        gait_trusted = gait_valid and gait_ev and (
            motion >= self.exit_motion or gait_ev_strong
        )

        # 镜头运动补偿：快速移动镜头时，静止人物的 bbox 中心位移会被污染成
        # "假行走"（motion 远超 enter_motion）。因此 walking 不再由位移单独触发，
        # 必须同时具备位移证据 + 步态/模型证据：
        #   - 步态证据可信（踝可见 + 有摆动幅度 + 换向节奏）
        #   - 或模型强烈支持 walking，且至少存在踝部摆动幅度（防止纯镜头抖动
        #     让 TCN 的 velocity 特征误判为 walking）
        want_walking = (
            motion_ev
            and (gait_trusted or (model_ev and amp >= self.gait_amp))
        )

        # 站立以 bbox 位移为准（最可靠信号），不再要求姿态能量/踝摆低于噪声地板
        want_standing = motion <= self.exit_motion

        if want_walking and want_standing:
            # 冲突时以更直接的位移证据为准
            want_standing = not (motion_ev or gait_trusted)

        desired = state["action"]
        if want_walking:
            desired = "walking"
        elif want_standing:
            desired = "standing"

        output_state = "valid"
        if desired != state["action"]:
            if state["candidate_action"] != desired:
                state["candidate_action"] = desired
                state["candidate_count"] = 1
            else:
                state["candidate_count"] += 1

            if state["candidate_count"] >= self.switch_streak:
                state["action"] = desired
                state["candidate_count"] = 0
            else:
                output_state = "held"
                desired = state["action"]
        else:
            state["candidate_action"] = desired
            state["candidate_count"] = 0

        if desired == "walking":
            score = min(
                0.96,
                0.58
                + 0.18 * (motion / max(self.enter_motion, 1e-6))
                + 0.14 * (pose_energy / max(self.enter_pose_energy, 1e-6))
                + 0.10 * min(1.0, amp / max(self.gait_amp, 1e-6)),
            )
            if model_action == "walking":
                score = max(score, model_score)
        else:
            still_level = max(
                motion / max(self.exit_motion, 1e-6),
                pose_energy / max(self.exit_pose_energy, 1e-6),
            )
            score = min(
                0.95,
                0.72 + 0.18 * (1.0 - min(1.0, still_level)),
            )
            if model_action == "standing":
                score = max(score, model_score)

        state["score"] = 0.70 * state["score"] + 0.30 * score
        out_state = (
            "warming"
            if state["samples"] < self.warmup_samples
            else output_state
        )
        return {
            "action": state["action"],
            "confidence": float(state["score"]),
            "state": out_state,
            "channel": "locomotion",
            "amp": round(amp, 4),
        }
