# ============================================================
# engine/geometry.py
#
# 姿态几何特征（纯 Python 标准库，无第三方依赖）
#
# 输入: 33 点 landmarks，每个点具有 .x / .y / .visibility（归一化坐标）
# 输出: 膝盖角度、躯干倾斜、躯干竖直分量等原始几何量 + 各量有效性标志，
#       以及步态证据（踝部摆动幅度、换向节奏）
#
# 供 locomotion_engine / special_action_engine 使用
#
# === VISIBILITY_MASK_PATCH (2026-09-30) ===
# 调研依据（WACV 2024《Rethinking Visibility in Human Pose Estimation》）：
#   把不可见关键点当有效数据参与计算，不仅无益，还会损害整体准确性；
#   正确做法是 visibility masking —— 不可见点必须被屏蔽。
#
# 实测问题（本工程，2026-09-30 用户反馈）：
#   脚部被遮挡时，踝关节坐标是 convert17to33 继承来的"假坐标"，
#   用它算出的膝角/踝摆完全是垃圾数据，却仍被下游判据采信 ——
#   导致"识别不到脚就判站立""坐着被判下蹲"。
#
# 修复方案：
#   ① 每个几何量附带 *_valid 标志：参与计算的关键点全部可见才为 True
#   ② ankle_amp/cadence 增加可见性过滤：不可见帧的样本直接丢弃
#   ③ compute_geometry 在躯干四点不全可见时返回 default + valid=False
# ============================================================

import math


EPS = 1e-6

# 关键点可见性阈值（与 KP_VIS_THRESHOLD / _visible_ratio 一致）
VIS_THRESHOLD = 0.3


def _visible(pt):
    """关键点是否可见（visibility 掩码核心判定）"""
    return getattr(pt, "visibility", 1.0) > VIS_THRESHOLD


def _angle_deg(ax, ay, bx, by, cx, cy):
    """计算 a-b-c 三点夹角（b 为顶点），单位度"""
    ba_x = ax - bx
    ba_y = ay - by
    bc_x = cx - bx
    bc_y = cy - by
    ba_norm = math.hypot(ba_x, ba_y)
    bc_norm = math.hypot(bc_x, bc_y)
    denom = ba_norm * bc_norm + EPS
    cos = (ba_x * bc_x + ba_y * bc_y) / denom
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def knee_angle(kpts, left=True):
    """髋-膝-踝夹角（MediaPipe/COCO 33 点索引）"""
    hip = 23 if left else 24
    knee = 25 if left else 26
    ankle = 27 if left else 28
    return _angle_deg(
        kpts[hip].x, kpts[hip].y,
        kpts[knee].x, kpts[knee].y,
        kpts[ankle].x, kpts[ankle].y,
    )


def knee_angle_valid(kpts, left=True):
    """膝角计算所需三点（髋/膝/踝）是否全部可见"""
    hip = 23 if left else 24
    knee = 25 if left else 26
    ankle = 27 if left else 28
    return (
        _visible(kpts[hip]) and _visible(kpts[knee]) and _visible(kpts[ankle])
    )


def torso_vector(kpts):
    """肩部中心 -> 髋部中心 向量 (dx, dy)"""
    sx = (kpts[11].x + kpts[12].x) * 0.5
    sy = (kpts[11].y + kpts[12].y) * 0.5
    hx = (kpts[23].x + kpts[24].x) * 0.5
    hy = (kpts[23].y + kpts[24].y) * 0.5
    return hx - sx, hy - sy


def torso_valid(kpts):
    """躯干向量所需四点（双肩双髋）是否全部可见"""
    return all(
        _visible(kpts[i]) for i in (11, 12, 23, 24)
    )


def torso_tilt(kpts):
    """躯干水平/垂直比：直立≈0，弯腰增大，跌倒(横向)很大"""
    dx, dy = torso_vector(kpts)
    return abs(dx) / max(abs(dy), EPS)


def hip_drop(kpts):
    """躯干竖直分量占比：直立≈1.0，横向(跌倒)趋近 0"""
    dx, dy = torso_vector(kpts)
    span = math.hypot(dx, dy) + EPS
    return abs(dy) / span


def compute_geometry(kpts):
    """
    输出原始几何量 + 有效性标志（阈值判断交给各通道引擎，便于标定）

    返回 dict 新增字段：
        knee_valid    : 至少一侧膝角计算可信（髋膝踝三点均可见）
        knee_both_valid: 双侧膝角均可信
        torso_valid   : 躯干倾斜/竖直分量可信（肩髋四点可见）
        foot_valid    : 至少一侧踝可见（重心/落位判据可用）
    """
    default = {
        "knee_left": 180.0,
        "knee_right": 180.0,
        "knee_mean": 180.0,
        "torso_tilt": 0.0,
        "hip_drop": 1.0,
        # 有效标志：无数据时全部为 False，下游判据据此跳过
        "knee_valid": False,
        "knee_both_valid": False,
        "torso_valid": False,
        "foot_valid": False,
    }
    if kpts is None or len(kpts) < 29:
        return default

    kv_l = knee_angle_valid(kpts, left=True)
    kv_r = knee_angle_valid(kpts, left=False)
    tv = torso_valid(kpts)
    fv = _visible(kpts[27]) or _visible(kpts[28])

    knee_l = knee_angle(kpts, left=True) if kv_l else 180.0
    knee_r = knee_angle(kpts, left=False) if kv_r else 180.0

    # 膝角均值：只用可信侧（双侧可信取均值，单侧可信取该侧）
    if kv_l and kv_r:
        knee_mean = (knee_l + knee_r) * 0.5
    elif kv_l:
        knee_mean = knee_l
    elif kv_r:
        knee_mean = knee_r
    else:
        knee_mean = 180.0

    return {
        "knee_left": knee_l,
        "knee_right": knee_r,
        "knee_mean": knee_mean,
        "torso_tilt": torso_tilt(kpts) if tv else 0.0,
        "hip_drop": hip_drop(kpts) if tv else 1.0,
        "knee_valid": kv_l or kv_r,
        "knee_both_valid": kv_l and kv_r,
        "torso_valid": tv,
        "foot_valid": fv,
    }


def ankle_amp(points):
    """
    步态证据1：左右踝 y 向摆动幅度（一段时间窗口内 max-min）

    === VISIBILITY_MASK_PATCH ===
    不可见帧的踝坐标是继承来的假数据，直接丢弃（原来无此过滤，
    实测垃圾值可达 0.17~4.9，引发"抖腿判行走"）。
    返回 (amp, n_samples)：n_samples 为有效样本数，不足 3 个时调用方应视为无证据。
    """
    ys_l = []
    ys_r = []
    for pts in points:
        if pts is None or len(pts) < 29:
            continue
        # 掩码：该帧踝可见才采信其坐标
        if _visible(pts[27]):
            ys_l.append(pts[27].y)
        if _visible(pts[28]):
            ys_r.append(pts[28].y)
    n = max(len(ys_l), len(ys_r))
    if n < 3:
        return 0.0, 0
    amp = max(
        max(ys_l) - min(ys_l) if len(ys_l) >= 3 else 0.0,
        max(ys_r) - min(ys_r) if len(ys_r) >= 3 else 0.0,
    )
    return amp, n


def cadence(points):
    """
    步态证据2：踝部 y 向运动换向次数（跨步节奏）

    === VISIBILITY_MASK_PATCH ===
    同样加可见性过滤：任一帧的踝不可见时跳过该相邻对（不构成换向证据）。
    """
    prev_sign_l = 0
    prev_sign_r = 0
    flips = 0
    prev_pts = None
    for cur_pts in points:
        if prev_pts is None:
            prev_pts = cur_pts
            continue
        if (
            prev_pts is None or cur_pts is None
            or len(prev_pts) < 29 or len(cur_pts) < 29
        ):
            prev_pts = cur_pts
            continue
        # 掩码：前后两帧的踝都可见才构成一对有效差分
        l_ok = _visible(cur_pts[27]) and _visible(prev_pts[27])
        r_ok = _visible(cur_pts[28]) and _visible(prev_pts[28])

        if l_ok:
            dy_l = cur_pts[27].y - prev_pts[27].y
            if dy_l > 0:
                sign_l = 1
            elif dy_l < 0:
                sign_l = -1
            else:
                sign_l = 0
            if sign_l != 0 and prev_sign_l != 0 and sign_l != prev_sign_l:
                flips += 1
            if sign_l != 0:
                prev_sign_l = sign_l
        if r_ok:
            dy_r = cur_pts[28].y - prev_pts[28].y
            if dy_r > 0:
                sign_r = 1
            elif dy_r < 0:
                sign_r = -1
            else:
                sign_r = 0
            if sign_r != 0 and prev_sign_r != 0 and sign_r != prev_sign_r:
                flips += 1
            if sign_r != 0:
                prev_sign_r = sign_r
        prev_pts = cur_pts
    return flips
