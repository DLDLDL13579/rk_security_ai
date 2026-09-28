"""
============================================================
test_multi_behavior_eval.py
多人行为实时评估工具（无头模式，适用于 RK3588 板端）

用法示例:
    python test_multi_behavior_eval.py                        # 默认跑 videos/ 全部视频
    python test_multi_behavior_eval.py --source videos
    python test_multi_behavior_eval.py --source videos/standing/stand_001.mp4
    python test_multi_behavior_eval.py --source rtsp://user:pass@ip:554/ch01 --save-video 1

说明:
    - --source 传目录时按子目录名(standing/walking/squat/bend/fall_down)作为标签，
      统计五类混淆矩阵；
    - 传视频文件/RTSP 时只输出实时统计（多人场景适用）；
    - 全程无 GUI，默认输出 output/eval_results.csv。

环境变量（均可覆盖命令行默认值）:
    RK_SOURCE            视频源（目录/文件/RTSP）
    RK_EVAL_OUT          CSV 输出路径
    RK_EVAL_SAVE_VIDEO   1 保存标注视频
    RK_EVAL_MAX_FRAMES   最多处理帧数（0=不限）
    RK_YOLO_MODEL / RK_POSE_MODEL / RK_BEHAVIOR_MODEL  模型路径
============================================================
"""

import argparse
import csv
import os
import sys
import time


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

CLASSES = ["standing", "walking", "squat", "bend", "fall_down"]


def resolve_model_path(env_name, default_rel):
    value = os.environ.get(env_name)
    if value:
        return value
    return os.path.join(PROJECT_ROOT, default_rel)


def resolve_source(value):
    """支持: 绝对路径 / rtsp(s):// / 相对路径(相对工程根) / 默认 videos"""
    if value is None:
        value = os.environ.get("RK_SOURCE") or "videos"
    if "://" in value or os.path.isabs(value):
        return value
    return os.path.join(PROJECT_ROOT, value)


def build_engine(pose_backend="mediapipe"):
    from npu.yolo_rknn_v2 import YOLO_RKNN
    from pose.mediapipe_pose import MediaPipePose
    from engine.engine import PoseEngine

    yolo = YOLO_RKNN(resolve_model_path(
        "RK_YOLO_MODEL", os.path.join("npu", "yolov5s-640-640.rknn")
    ))
    if pose_backend == "rknn":
        from npu.pose_rknn import PoseRKNN
        pose = PoseRKNN(resolve_model_path(
            "RK_POSE_MODEL", os.path.join("npu", "model.rknn")
        ))
    else:
        pose = MediaPipePose()
    engine = PoseEngine(
        yolo=yolo,
        pose=pose,
        behavior_model_path=resolve_model_path(
            "RK_BEHAVIOR_MODEL",
            os.path.join("npu", "onnx-rknn", "behavior_tcn.rknn"),
        ),
    )
    return engine


def run_video(engine, source, gt_label, out_csv, save_video, max_frames):
    """处理一路视频/RTSP，返回统计摘要；逐帧结果写入 CSV。"""
    import cv2

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[SKIP] cannot open:", source)
        return None

    writer = None
    frame_index = 0
    start = time.time()
    tracks_seen = {}
    frame_rows = []
    det_active = {}
    beh_active = {}
    det_rows = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if max_frames and frame_index > max_frames:
                break

            det_result = engine.process_detections(frame)
            beh_result = engine.process_behaviors(
                frame, frame_id=frame_index, tracks=det_result.get("detections", [])
            )

            # 按检测框统计每帧活跃人数（行为结果可能滞后，不能代表真实并发）
            det_ids = [
                getattr(d, "track_id", -1)
                for d in det_result.get("detections", [])
            ]
            if det_ids:
                det_active.setdefault(frame_index, set()).update(det_ids)

            if os.environ.get("RK_EVAL_DEBUG_DET") == "1" and frame_index <= 40:
                boxes = [
                    getattr(d, "bbox", None)
                    for d in det_result.get("detections", [])
                ]
                print(
                    "[DET] frame=%d n=%d boxes=%s"
                    % (frame_index, len(boxes), boxes)
                )

            # 每帧检测框写入 CSV（"det" 行），无需抓终端即可远程定位检测问题
            det_ids_str = ",".join(str(t) for t in det_ids)
            boxes_str = ";".join(
                ",".join(str(int(v)) for v in getattr(d, "bbox", []))
                for d in det_result.get("detections", [])
            )
            out_csv.writerow([
                "det", os.path.basename(source), frame_index,
                len(det_result.get("detections", [])), det_ids_str, boxes_str,
            ])
            det_rows.append([
                frame_index,
                [
                    (getattr(d, "track_id", -1), list(getattr(d, "bbox", [])))
                    for d in det_result.get("detections", [])
                ],
            ])

            for b in beh_result.get("behaviors", []):
                tid = b.get("track_id", -1)
                beh_active.setdefault(frame_index, set()).add(tid)
                action = b.get("action", "unknown")
                state = b.get("state", "valid")
                conf = float(b.get("confidence", 0.0))

                st = tracks_seen.setdefault(tid, {
                    "first_frame": frame_index,
                    "first_valid_frame": None,
                    "last_action": None,
                    "flips": 0,
                    "frames": 0,
                    "action_votes": {},
                    "valid_votes": {},
                })
                st["frames"] += 1
                st["action_votes"][action] = st["action_votes"].get(action, 0) + 1
                if state == "valid":
                    st["valid_votes"][action] = st["valid_votes"].get(action, 0) + 1
                if st["last_action"] is not None and action != st["last_action"]:
                    st["flips"] += 1
                st["last_action"] = action
                if state == "valid" and st["first_valid_frame"] is None:
                    st["first_valid_frame"] = frame_index

                frame_rows.append([
                    os.path.basename(source), frame_index, tid,
                    action, round(conf, 3), state,
                    round(float(b.get("motion", 0.0)), 4),
                    round(float(b.get("pose_energy", 0.0)), 4),
                    round(float(b.get("gait_amp", 0.0)), 4),
                ])

            if save_video:
                for b in beh_result.get("behaviors", []):
                    bbox = b.get("bbox")
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    color = (0, 0, 255) if b.get("action") == "fall_down" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = "ID{} {}".format(
                        b.get("track_id", -1), b.get("action", "unknown")
                    )
                    cv2.putText(
                        frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                    )
                if writer is None:
                    out_dir = os.path.join(PROJECT_ROOT, "output")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(
                        out_dir, "eval_{}.mp4".format(
                            os.path.splitext(os.path.basename(source))[0]
                        )
                    )
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h)
                    )
                writer.write(frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = max(1e-6, time.time() - start)
    fps = frame_index / elapsed
    det_frames = len(det_active)
    beh_frames = len(beh_active)
    max_concurrent = max((len(v) for v in det_active.values()), default=0)
    coverage = round(beh_frames / max(1, det_frames), 3)

    summary = {
        "source": os.path.basename(source),
        "gt": gt_label,
        "frames": frame_index,
        "fps": round(fps, 2),
        "tracks": len(tracks_seen),
        "max_concurrent": max_concurrent,
        "coverage": coverage,
    }

    latencies = []
    flip_counts = []
    vote_weights = {}
    for tid, st in tracks_seen.items():
        if st["first_valid_frame"] is not None:
            latencies.append(st["first_valid_frame"] - st["first_frame"])
        flip_counts.append(st["flips"])
        # 仅统计有效帧 ≥15 的轨迹，避免 warming 短轨碎片干扰预测
        if st["frames"] < 15:
            continue
        votes = st["valid_votes"] if st["valid_votes"] else st["action_votes"]
        for action, cnt in votes.items():
            vote_weights[action] = vote_weights.get(action, 0) + cnt

    summary["first_valid_latency_frames"] = (
        min(latencies) if latencies else -1
    )
    summary["avg_flips_per_track"] = (
        round(sum(flip_counts) / max(1, len(flip_counts)), 2)
    )
    summary["pred"] = (
        max(vote_weights, key=vote_weights.get)
        if vote_weights else "unknown"
    )

    out_csv.writerow(["summary"] + [summary[k] for k in [
        "source", "gt", "frames", "fps", "tracks", "max_concurrent",
        "coverage", "first_valid_latency_frames", "avg_flips_per_track", "pred",
    ]])
    for row in frame_rows:
        out_csv.writerow(["frame"] + row)

    print("=" * 64)
    print("VIDEO :", summary["source"], "(gt=%s)" % summary["gt"])
    print("FPS   :", summary["fps"])
    print("TRACK : total=%d max_concurrent=%d" % (
        summary["tracks"], summary["max_concurrent"]
    ))
    print("COVER : 行为覆盖率=%.1f%%（行为帧/检测帧）" % (summary["coverage"] * 100))
    print("LATENCY(first valid, frames):", summary["first_valid_latency_frames"])
    print("STABLE(avg action flips/track):", summary["avg_flips_per_track"])
    print("PRED  :", summary["pred"])

    # 每人行为分布
    for tid, st in sorted(tracks_seen.items()):
        dist = dict(sorted(st["action_votes"].items(), key=lambda x: -x[1]))
        print(f"  ID{tid} 行为分布: {dist} (frames={st['frames']}, flips={st['flips']})")

    # 每人检测框平均位移/身高（判断站立视频是否有真实运动）
    prev_pos = {}
    track_speed = {}
    for frm, items in det_rows:
        for tid, box in items:
            if len(box) < 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            h = max(1.0, box[3] - box[1])
            if tid in prev_pos:
                pcx, pcy, ph, pf = prev_pos[tid]
                dt = max(1, frm - pf)
                dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
                speed = dist / max(1.0, (ph + h) / 2.0) / dt
                track_speed.setdefault(tid, []).append(speed)
            prev_pos[tid] = (cx, cy, h, frm)
    for tid in sorted(track_speed):
        avg_speed = sum(track_speed[tid]) / max(1, len(track_speed[tid]))
        print(f"  ID{tid} 平均位移/身高: {avg_speed:.4f} (帧数={len(track_speed[tid])})")

    print("=" * 64)
    return summary


def evaluate_directory(engine, src_dir, out_csv, save_video, max_frames):
    conf_matrix = {}
    for gt in CLASSES:
        folder = os.path.join(src_dir, gt)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".mp4"):
                continue
            summary = run_video(
                engine, os.path.join(folder, name), gt,
                out_csv, save_video, max_frames,
            )
            if summary is None:
                continue
            conf_matrix.setdefault(gt, {})[summary["pred"]] = (
                conf_matrix.get(gt, {}).get(summary["pred"], 0) + 1
            )

    print("\n========== Confusion Matrix ==========")
    header = ["gt \\ pred"] + CLASSES + ["acc"]
    print("  ".join("{:<10}".format(h) for h in header))
    for gt in CLASSES:
        if gt not in conf_matrix:
            continue
        row = conf_matrix[gt]
        total = sum(row.values())
        acc = row.get(gt, 0) / max(1, total)
        cells = ["{:<10}".format(gt)]
        for pred in CLASSES:
            cells.append("{:<10}".format(row.get(pred, 0)))
        cells.append("{:<10}".format(round(acc, 2)))
        print("  ".join(cells))


def main():
    parser = argparse.ArgumentParser(description="multi-person behavior eval")
    parser.add_argument(
        "--source", nargs="?", default=None,
        help="video dir / file / rtsp url (default: RK_SOURCE or videos/)",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get(
            "RK_EVAL_OUT",
            os.path.join(PROJECT_ROOT, "output", "eval_results.csv"),
        ),
    )
    parser.add_argument(
        "--save-video", type=int,
        default=int(os.environ.get("RK_EVAL_SAVE_VIDEO", "0")),
    )
    parser.add_argument(
        "--max-frames", type=int,
        default=int(os.environ.get("RK_EVAL_MAX_FRAMES", "0")),
    )
    parser.add_argument(
        "--pose-backend",
        default=os.environ.get("RK_POSE_BACKEND", "mediapipe"),
        choices=["mediapipe", "rknn"],
    )
    args = parser.parse_args()

    source = resolve_source(args.source)
    if args.source is None:
        print("[EVAL] no --source, use default:", source)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        out_csv = csv.writer(f)
        out_csv.writerow(["kind", "source", "gt", "frames", "fps", "tracks",
                          "max_concurrent", "coverage",
                          "latency_frames", "avg_flips", "pred"])

        engine = build_engine(pose_backend=args.pose_backend)
        if os.path.isdir(source):
            evaluate_directory(engine, source, out_csv, args.save_video, args.max_frames)
        else:
            run_video(
                engine, source, "unknown",
                out_csv, args.save_video, args.max_frames,
            )

    print("\nCSV saved:", args.out)


if __name__ == "__main__":
    main()
