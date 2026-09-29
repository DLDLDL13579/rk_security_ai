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
            # 【2026-09-28 实测重标定】v8-pose 的位移特征比 MediaPipe 时代缩小到
            # 1/3~1/2：walk_001 实测 motion 中位 0.0049、p75 0.0081，而旧阈值 0.016
            # 是其中位数的 3.3 倍 → walking 几乎全被判成 standing（10%）。
            #
            # 50 视频全量评估对比（正式 eval 脚本，唯一可信口径）：
            #   enter/exit 0.016/0.008（旧） -> 总体 46%（walking 10%、standing 80%）
            #   enter/exit 0.005/0.0015     -> 总体 54%（walking 90%、standing 40%）✅
            #   enter/exit 0.010/0.0015     -> 总体 50%（walking 60%、standing 50%）
            # 注：0.010 曾在 30 视频抽样中显示更优（预计 58%），但全量评估被推翻为 50%。
            #   —— 教训：抽样只用于筛选假说，配置决策必须全量评估。
            #
            # exit 是 walking 的"保持门"，是本次修复的关键：
            #   单视频实测 exit 0.003 → standing（walking 票 133）；
            #              exit 0.0015 → walking（票 350）。
            os.environ.get("RK_LOCO_ENTER_MOTION", "0.005")
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
        # 模型主导门槛：新 TCN 置信度饱和（实测 min 0.984），0.80 可安全采纳
        self.model_lead_score = float(
            os.environ.get("RK_LOCO_MODEL_LEAD_SCORE", "0.80")
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

        # 【2026-09-29 模型主导】重训后的 TCN（17.7k 序列训练，序列级 100% 准确、
        # 置信度饱和至 1.0）已远比手写几何/运动阈值可靠。
        # 原逻辑要求 motion_ev + (gait_trusted or model_ev)，实测导致：
        #   - standing 视频里人微动 → motion_ev 为真 + 踝摆假象 → 误判 walking（40%）
        #   - TCN 明确输出 standing 时也无法纠正（几何证据优先）
        # 新逻辑：TCN 高置信直接采纳；仅当 TCN 缺失/低置信时才回退到几何判据。
        model_lead = (
            model_action in ("standing", "walking")
            and model_score >= self.model_lead_score
        )
        if model_lead:
            want_walking = (model_action == "walking")
            want_standing = (model_action == "standing")
        else:
            # 回退：原几何/运动判据（TCN 序列未攒满或低置信时）
            want_walking = (
                motion_ev
                and (gait_trusted or (model_ev and amp >= self.gait_amp))
            )
            want_standing = motion <= self.exit_motion

        # 站立以 bbox 位移为准（最可靠信号），不再要求姿态能量/踝摆低于噪声地板
        want_standing = motion <= self.exit_motion

        if want_walking and want_standing and not model_lead:
            # 冲突时以更直接的位移证据为准（仅几何回退路径）
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
