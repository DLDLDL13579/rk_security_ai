# ============================================================
# engine/geometry.py
#
# 姿态几何特征（纯 Python 标准库，无第三方依赖）
#
# 输入: 33 点 landmarks，每个点具有 .x / .y（归一化坐标）
# 输出: 膝盖角度、躯干倾斜、躯干竖直分量等原始几何量，
#       以及步态证据（踝部摆动幅度、换向节奏）
#
# 供 locomotion_engine / special_action_engine 使用
# ============================================================

import math


EPS = 1e-6


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


def torso_vector(kpts):
    """肩部中心 -> 髋部中心 向量 (dx, dy)"""
    sx = (kpts[11].x + kpts[12].x) * 0.5
    sy = (kpts[11].y + kpts[12].y) * 0.5
    hx = (kpts[23].x + kpts[24].x) * 0.5
    hy = (kpts[23].y + kpts[24].y) * 0.5
    return hx - sx, hy - sy


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
    """输出原始几何量（阈值判断交给各通道引擎，便于标定）"""
    default = {
        "knee_left": 180.0,
        "knee_right": 180.0,
        "knee_mean": 180.0,
        "torso_tilt": 0.0,
        "hip_drop": 1.0,
    }
    if kpts is None or len(kpts) < 29:
        return default
    knee_l = knee_angle(kpts, left=True)
    knee_r = knee_angle(kpts, left=False)
    return {
        "knee_left": knee_l,
        "knee_right": knee_r,
        "knee_mean": (knee_l + knee_r) * 0.5,
        "torso_tilt": torso_tilt(kpts),
        "hip_drop": hip_drop(kpts),
    }


def ankle_amp(points):
    """步态证据1：左右踝 y 向摆动幅度（一段时间窗口内 max-min）"""
    if not points:
        return 0.0
    ys_l = []
    ys_r = []
    for pts in points:
        if pts is None or len(pts) < 29:
            continue
        ys_l.append(pts[27].y)
        ys_r.append(pts[28].y)
    if not ys_l:
        return 0.0
    return max(max(ys_l) - min(ys_l), max(ys_r) - min(ys_r))


def cadence(points):
    """步态证据2：踝部 y 向运动换向次数（跨步节奏）"""
    if len(points) < 3:
        return 0
    prev_sign_l = 0
    prev_sign_r = 0
    flips = 0
    for i in range(1, len(points)):
        prev_pts = points[i - 1]
        cur_pts = points[i]
        if (
            prev_pts is None or cur_pts is None
            or len(prev_pts) < 29 or len(cur_pts) < 29
        ):
            continue
        dy_l = cur_pts[27].y - prev_pts[27].y
        dy_r = cur_pts[28].y - prev_pts[28].y
        if dy_l > 0:
            sign_l = 1
        elif dy_l < 0:
            sign_l = -1
        else:
            sign_l = 0
        if dy_r > 0:
            sign_r = 1
        elif dy_r < 0:
            sign_r = -1
        else:
            sign_r = 0

        if sign_l != 0 and prev_sign_l != 0 and sign_l != prev_sign_l:
            flips += 1
        if sign_r != 0 and prev_sign_r != 0 and sign_r != prev_sign_r:
            flips += 1
        if sign_l != 0:
            prev_sign_l = sign_l
        if sign_r != 0:
            prev_sign_r = sign_r
    return flips
