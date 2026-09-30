# -*- coding: utf-8 -*-
"""
============================================================
engine/special_action_engine.py

Special action 通道：仅负责 squat / bend / fall_down。

以几何候选为主判据（膝盖角度、躯干倾斜、躯干竖直分量、bbox 宽高比），
TCN 置信度作为加速确认的先验，避免五分类模型边界互相挤压。

几何阈值可通过 RK_SPECIAL_* 环境变量标定。
============================================================
"""

import os
import time
from collections import deque


def _now():
    """POST_FALL_PHASE_PATCH 用：单调时钟（不受系统改时间影响）"""
    return time.monotonic()


class SpecialActionEngine:

    def __init__(self):
        # TCN 先验阈值（模型强支持时可加速确认）
        self.score_thresholds = {
            "squat": float(os.environ.get("RK_SPECIAL_SQUAT_MIN_SCORE", "0.55")),
            "bend": float(os.environ.get("RK_SPECIAL_BEND_MIN_SCORE", "0.55")),
            "fall_down": float(os.environ.get("RK_SPECIAL_FALL_MIN_SCORE", "0.70")),
        }
        # 模型主导门槛：新 TCN 置信度饱和（实测 min 0.984），0.80 可安全采纳
        self.model_lead_score = float(
            os.environ.get("RK_SPECIAL_MODEL_LEAD_SCORE", "0.80")
        )
        self._unused_thresholds = {
        }
        # 模型支持时的确认帧数
        self.streaks = {
            "squat": max(1, int(os.environ.get("RK_SPECIAL_SQUAT_STREAK", "2"))),
            "bend": max(1, int(os.environ.get("RK_SPECIAL_BEND_STREAK", "2"))),
            "fall_down": max(1, int(os.environ.get("RK_SPECIAL_FALL_STREAK", "3"))),
        }
        # 纯几何（无模型支持）时的确认帧数
        self.geo_streaks = {
            "squat": max(2, int(os.environ.get("RK_SPECIAL_SQUAT_GEO_STREAK", "5"))),
            "bend": max(2, int(os.environ.get("RK_SPECIAL_BEND_GEO_STREAK", "4"))),
            "fall_down": max(3, int(os.environ.get("RK_SPECIAL_FALL_GEO_STREAK", "5"))),
        }
        self.min_pose_quality = float(
            os.environ.get("RK_SPECIAL_MIN_POSE_QUALITY", "0.28")
        )
        self.fall_pose_quality = float(
            os.environ.get("RK_SPECIAL_FALL_POSE_QUALITY", "0.36")
        )

        # 几何阈值
        self.squat_knee = float(
            os.environ.get("RK_SPECIAL_SQUAT_KNEE", "95.0")
        )
        self.squat_tilt = float(
            os.environ.get("RK_SPECIAL_SQUAT_TILT", "0.45")
        )
        self.bend_tilt_lo = float(
            os.environ.get("RK_SPECIAL_BEND_TILT_LO", "0.40")
        )
        self.bend_tilt_hi = float(
            os.environ.get("RK_SPECIAL_BEND_TILT_HI", "1.15")
        )
        self.bend_knee = float(
            os.environ.get("RK_SPECIAL_BEND_KNEE", "115.0")
        )
        self.fall_tilt = float(
            os.environ.get("RK_SPECIAL_FALL_TILT", "0.75")
        )
        self.fall_drop = float(
            os.environ.get("RK_SPECIAL_FALL_DROP", "0.62")
        )
        self.fall_bbox_ratio = float(
            os.environ.get("RK_SPECIAL_FALL_BBOX_RATIO", "1.00")
        )
        self.fall_height_drop = float(
            os.environ.get("RK_SPECIAL_FALL_HEIGHT_DROP", "1.45")
        )
        self.hold_frames = max(
            0,
            int(os.environ.get("RK_SPECIAL_HOLD_FRAMES", "6")),
        )
        # === POST_FALL_PHASE_PATCH (2026-09-30) ===
        # 后跌倒阶段确认时长：帧数口径（与视频帧率挂钩，评估/实时一致）
        # 视频 25fps × 3 秒 = 75 帧；墙钟秒数仅作 frame_id 缺失时的兜底
        self.fall_confirm_seconds = float(
            os.environ.get("RK_SPECIAL_FALL_CONFIRM_SECONDS", "3.0")
        )
        self.fall_confirm_frames = max(
            3,
            int(
                os.environ.get(
                    "RK_SPECIAL_FALL_CONFIRM_FRAMES",
                    str(int(round(self.fall_confirm_seconds * 25))),
                )
            ),
        )
        # 后跌倒阶段状态机: track_id -> {"phase","since","hits","misses"}
        self.post_fall = {}
        self.states = {}

        print("[SpecialAction] Ready (geometry + TCN prior)")

    def clear_track(self, track_id):
        self.states.pop(track_id, None)
        self.post_fall.pop(track_id, None)

    def _geometry_candidate(self, geometry, bbox):
        if geometry is None:
            return None
        bbox_w = max(1.0, float(bbox[2] - bbox[0]))
        bbox_h = max(1.0, float(bbox[3] - bbox[1]))
        bbox_ratio = bbox_w / bbox_h

        # === VISIBILITY_MASK_PATCH (2026-09-30) ===
        # squat/bend 依赖膝角 → 要求膝角可信（髋膝踝可见），否则跳过：
        # 用户实测"识别不到脚就被判下蹲"的直接根因。
        knee_ok = geometry.get("knee_valid", False)
        torso_ok = geometry.get("torso_valid", False)

        if (
            torso_ok
            and geometry.get("torso_tilt", 0.0) >= self.fall_tilt
            and geometry.get("hip_drop", 1.0) <= self.fall_drop
            and bbox_ratio >= self.fall_bbox_ratio
        ):
            return "fall_down"
        if (
            knee_ok
            and geometry.get("knee_mean", 180.0) <= self.squat_knee
            and torso_ok
            and geometry.get("torso_tilt", 0.0) <= self.squat_tilt
        ):
            return "squat"
        if (
            torso_ok
            and self.bend_tilt_lo
            <= geometry.get("torso_tilt", 0.0)
            <= self.bend_tilt_hi
            and knee_ok
            and geometry.get("knee_mean", 180.0) >= self.bend_knee
        ):
            return "bend"
        return None

    def _geometry_partial(self, geometry, bbox):
        """半匹配候选：遮挡/非标准动作下放宽阈值，输出动作（后续标记 partial）。
        仅覆盖 squat/bend；fall 保持严格（必须完整几何，防误报）。

        === VISIBILITY_MASK_PATCH (2026-09-30) ===
        partial 是放宽路径，同样要求膝角/躯干的基本可信，否则放宽无意义
        （放宽建立在垃圾数据上 = 误报制造机）。"""
        if geometry is None:
            return None
        tilt = geometry.get("torso_tilt", 0.0)
        knee = geometry.get("knee_mean", 180.0)
        knee_ok = geometry.get("knee_valid", False)
        torso_ok = geometry.get("torso_valid", False)
        if not (knee_ok or torso_ok):
            return None
        if knee_ok and knee <= self.squat_knee * 1.30 and torso_ok and tilt <= self.squat_tilt * 1.35:
            return "squat"
        if (
            torso_ok
            and tilt >= self.bend_tilt_lo * 0.75
            and tilt <= self.bend_tilt_hi * 1.15
            and knee_ok
            and knee >= self.bend_knee * 0.88
        ):
            return "bend"
        return None

    def resolve(
        self,
        track_id,
        action="",
        score=0.0,
        pose_quality=0.0,
        bbox=None,
        keypoints=None,
        geometry=None,
        frame_id=None,
    ):
        if bbox is None:
            bbox = [0, 0, 1, 1]

        if pose_quality < self.min_pose_quality:
            prev = self.states.get(track_id)
            if (
                prev is not None
                and prev.get("confirmed")
                and prev.get("hold_left", 0) > 0
            ):
                prev["hold_left"] -= 1
                return {
                    "action": prev["action"],
                    "confidence": self._confidence(prev["action"], 0.0),
                    "state": "held",
                    "channel": "special",
                }
            self.states.pop(track_id, None)
            return None

        # ---- 跌倒证据：几何 + 时序高度骤降（bbox 突然变矮且变宽）----
        prev_state = self.states.get(track_id)
        h_hist = prev_state.get("h_hist") if prev_state else None
        if h_hist is None:
            h_hist = deque(maxlen=10)
        bbox_h = max(1.0, float(bbox[3] - bbox[1]))
        bbox_w = max(1.0, float(bbox[2] - bbox[0]))
        bbox_ratio = bbox_w / bbox_h
        h_hist.append(bbox_h)

        # === VISIBILITY_MASK_PATCH (2026-09-30) ===
        # 躯干四点（肩髋）不可见时 torso_tilt/hip_drop 是假数据 → 不得作为
        # 跌倒几何证据（原来不查，遮挡时误判跌倒）
        torso_ok = geometry is not None and geometry.get("torso_valid", False)
        fall_geo = (
            torso_ok
            and geometry.get("torso_tilt", 0.0) >= self.fall_tilt
            and geometry.get("hip_drop", 1.0) <= self.fall_drop
            and bbox_ratio >= self.fall_bbox_ratio
        )
        fall_temporal = (
            len(h_hist) >= 5
            and bbox_h > 1.0
            and (sorted(h_hist)[len(h_hist) // 2] / bbox_h) >= self.fall_height_drop
            and bbox_ratio >= 0.85
        )
        # 瞬态跌倒特征（用于进入"候选"状态，尚不确认）
        fall_instant = fall_geo or fall_temporal

        # === POST_FALL_PHASE_PATCH (2026-09-30) ===
        # 调研依据（Igual et al. 2013 综述）：跌倒检测须区分"只检测冲击"与
        # "同时检测后跌倒阶段"。坐下/蹲下/弯腰都是瞬态——特征命中后人会
        # 重新起身/稳定；真实跌倒后人是【持续躺地不起】。
        # 用户实测痛点：坐着办公被反复误判跌倒（88 次/30 分钟）。
        # 方案：跌倒不再单帧/短窗确认，改为状态机：
        #   PHASE_CANDIDATE: 瞬态特征命中 → 开始计时
        #   PHASE_ONGOING  : 特征持续满足（躺地状态保持）→ 计满确认时长才确认
        #   立起/特征消失   → 清零回 idle
        #
        # 【计时口径：帧号差而非墙钟】——2026-09-30 评估实测教训：
        # 首次实现用 time.monotonic() 计 3 秒，但离线评估是【全速播放】，
        # 20 秒的视频几秒就跑完 → 墙钟永远到不了 3 秒 → 跌倒全部漏报
        # （SE 从 80% 崩到 30%）。改为按 frame_id 差值计时：视频 25fps，
        # 3 秒 = 75 帧，评估与实时下含义一致。
        pf = self.post_fall.setdefault(
            track_id, {"phase": "idle", "since": None, "hits": 0, "misses": 0}
        )
        if fall_instant:
            if pf["phase"] != "ongoing":
                pf["phase"] = "candidate"
                if pf["since"] is None:
                    pf["since"] = frame_id if frame_id else 0
                    pf["since_wall"] = _now()
                pf["hits"] += 1
            # ongoing / candidate 状态下持续命中都累计
            if pf["phase"] in ("candidate", "ongoing"):
                # 帧号差计时（主）+ 墙钟兜底（frame_id 缺失时退化）
                base = pf.get("since")
                if base:
                    elapsed_frames = max(0, (frame_id or 0) - base)
                    enough = elapsed_frames >= self.fall_confirm_frames
                else:
                    wall = pf.get("since_wall") or _now()
                    enough = (_now() - wall) >= self.fall_confirm_seconds
                if enough:
                    pf["phase"] = "ongoing"
                    pf["misses"] = 0
        else:
            # 特征消失：若从未到 ongoing 则回 idle（瞬态=坐下/弯腰，不是跌倒）
            if pf["phase"] == "candidate":
                pf["phase"] = "idle"
                pf["since"] = None
                pf["hits"] = 0
            elif pf["phase"] == "ongoing":
                # 已确认的跌倒，特征短暂消失（遮挡）时保持 1 次宽容
                pf["misses"] = pf.get("misses", 0) + 1
                if pf["misses"] > 1:
                    pf["phase"] = "idle"
                    pf["since"] = None

        fall_confirmed = pf["phase"] == "ongoing"

        geo_candidate = self._geometry_candidate(geometry, bbox)
        if geo_candidate is None and fall_confirmed:
            geo_candidate = "fall_down"
        model_action = action if action in self.score_thresholds else ""
        model_ok = (
            bool(model_action)
            and float(score) >= self.score_thresholds[model_action]
        )

        partial_candidate = None
        if geo_candidate is None:
            partial_candidate = self._geometry_partial(geometry, bbox)

        # 【2026-09-29 模型优先】重训后的 TCN（17.7k 序列，序列级 100% 准确）
        # 已远比手写几何规则可靠。原逻辑是"几何命中优先、模型兜底"（geo_candidate
        # 先于 model_ok），导致模型正确输出被几何误判覆盖。
        # 新逻辑：模型高置信优先；模型缺失/低置信时才回退几何判据。
        model_lead = model_ok and float(score) >= self.model_lead_score

        candidate = None
        is_partial = False
        if model_lead:
            candidate = model_action
        elif geo_candidate is not None:
            candidate = geo_candidate
        elif model_ok:
            candidate = model_action
        elif partial_candidate is not None:
            candidate = partial_candidate
            is_partial = True

        # === POST_FALL_PHASE_PATCH (2026-09-30) ===
        # 模型主导的 fall_down 同样要过后跌倒阶段门：
        # 重训后的 TCN 对"坐下/躺椅子"会输出饱和置信度的 fall_down（实测
        # walking 0.96~1.00 之间紧邻一次 fall_down 0.99 的抖动，以及坐着
        # 办公时 88 次误报）——单帧模型结论不可作为跌倒的充分证据。
        # 只有"持续躺地 ≥ fall_confirm_seconds"才确认跌倒。
        if candidate == "fall_down" and not fall_confirmed:
            # 特征已命中但还没持续够确认时长 → 不输出（避免坐下/弯腰误报）
            # 模型高置信 + 已持续 40% 时长 → 允许提前确认（模型与几何双证据
            # 都指向跌倒，且已持续约 1.2 秒）
            base = pf.get("since")
            if base:
                elapsed = max(0, (frame_id or 0) - base)
                early_ok = model_lead and elapsed >= self.fall_confirm_frames * 0.4
            else:
                wall = pf.get("since_wall") or _now()
                early_ok = (
                    model_lead
                    and (_now() - wall) >= self.fall_confirm_seconds * 0.4
                )
            if not early_ok:
                self.states.pop(track_id, None)
                return None

        # 几何门仅对"非模型主导"的 fall_down 生效（模型主导时信任模型输出）
        if candidate == "fall_down" and not model_lead and geo_candidate != "fall_down":
            self.states.pop(track_id, None)
            return None
        if candidate == "fall_down" and pose_quality < self.fall_pose_quality:
            self.states.pop(track_id, None)
            return None

        if candidate is None:
            prev = self.states.get(track_id)
            if (
                prev is not None
                and prev.get("confirmed")
                and prev.get("hold_left", 0) > 0
            ):
                prev["hold_left"] -= 1
                return {
                    "action": prev["action"],
                    "confidence": self._confidence(prev["action"], 0.0),
                    "state": "held",
                    "channel": "special",
                }
            self.states.pop(track_id, None)
            return None

        require_geo = candidate != model_action or not model_ok
        needed = (
            self.geo_streaks[candidate]
            if require_geo
            else self.streaks[candidate]
        )
        if is_partial:
            needed += 2

        state = self.states.get(track_id)
        if state is None or state.get("candidate") != candidate:
            state = {
                "candidate": candidate,
                "count": 1,
                "confirmed": False,
                "action": None,
                "hold_left": 0,
                "h_hist": h_hist,
            }
        else:
            state["count"] += 1
            state["h_hist"] = h_hist
        self.states[track_id] = state

        if state["confirmed"]:
            state["hold_left"] = self.hold_frames
            return {
                "action": candidate,
                "confidence": self._confidence(
                    candidate, score if model_ok else 0.0
                ),
                "state": "partial" if is_partial else "valid",
                "channel": "special",
            }

        if state["count"] < needed:
            return {
                "action": candidate,
                "confidence": self._confidence(candidate, 0.0),
                "state": "warming",
                "channel": "special",
            }

        state["confirmed"] = True
        state["action"] = candidate
        state["hold_left"] = self.hold_frames
        return {
            "action": candidate,
            "confidence": self._confidence(
                candidate, score if model_ok else 0.0
            ),
            "state": "partial" if is_partial else "valid",
            "channel": "special",
        }

    @staticmethod
    def _confidence(action, model_score):
        geo_conf = {
            "squat": 0.72,
            "bend": 0.70,
            "fall_down": 0.88,
        }.get(action, 0.70)
        return float(max(geo_conf, model_score))
