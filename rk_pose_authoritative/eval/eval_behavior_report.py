# -*- coding: utf-8 -*-
"""
============================================================
eval_behavior_report.py

行为识别评估报告生成器（在 test_multi_behavior_eval.py 的 CSV 基础上增强）

为什么需要它（调研结论，2026-09-30）：
    Igual et al. 的跌倒检测综述（Biomed Eng Online 2013）指出，跌倒检测系统
    必须用 **敏感度 SE / 特异度 SP** 双指标衡量：SE = 真实跌倒被判为跌倒的比例，
    SP = 非跌倒活动（ADL）被判为非跌倒的比例。

    而原有评估脚本只输出「5 类混淆矩阵 + 总体精度」——跌倒误报会被平均掉，
    无法暴露「无人跌倒却反复报跌倒」这类致命问题（实测 60 秒误报 4 次，
    在混淆矩阵里完全看不见）。

本脚本输出：
    1. 五类混淆矩阵 + 每类 precision / recall / F1
    2. **跌倒敏感度 SE**（fall_down 视频被判为 fall_down 的比例）
    3. **跌倒特异度 SP**（非跌倒视频中被误判为 fall_down 的比例 → 误报率）
    4. 每类的误报率（该类视频被判成其他类的比例）
    5. 逐视频明细（便于定位是哪些视频拖后腿）

用法（板端）：
    python eval/eval_behavior_report.py --csv /tmp/eval_xxx.csv
    python eval/eval_behavior_report.py --csv a.csv b.csv   # 多份对比
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict


CLASSES = ["standing", "walking", "squat", "bend", "fall_down"]


def read_summaries(path):
    """从评估 CSV 读回 (gt, pred) 对"""
    pairs = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) >= 3 and row[0] == "summary":
                # 表头顺序: source, gt, frames, ...
                src, gt = row[1], row[2]
                # pred 在最后
                pred = row[-1]
                pairs.append((src, gt, pred))
    return pairs


def read_event_level(path):
    """
    事件级口径统计（2026-09-30 新增，反映安防系统真实价值）

    为什么需要它：
        逐帧明细（frame 行）记录每帧的行为标签。对跌倒检测而言，真正重要的是
        「这段视频里是否报出过跌倒」——而不是「跌倒帧数是否多于走动帧数」。
        实测发现投票口径会把「走向镜头后跌倒」的视频判成 walking（走动帧
        414 帧 vs 倒地 48 帧），从而严重低估跌倒检测能力。
        Igual et al. 综述亦要求以 SE/SP 衡量，而非整体投票精度。

    返回: {video: {"gt": gt, "counts": {action: n}, "fall_frames": n}}
    """
    data = {}
    cur = None
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            if row[0] == "summary" and len(row) > 2:
                cur = row[1]
                data.setdefault(cur, {"gt": row[2], "counts": {}, "fall_frames": 0})
            elif row[0] == "frame" and cur and len(row) >= 5:
                action = row[4]
                rec = data[cur]
                rec["counts"][action] = rec["counts"].get(action, 0) + 1
                if action == "fall_down":
                    rec["fall_frames"] += 1
    return data


def print_event_level(title, path):
    """事件级 SE/SP 报告"""
    data = read_event_level(path)
    if not data:
        return None

    falls = [v for v in data.values() if v["gt"] == "fall_down"]
    adls = [v for v in data.values() if v["gt"] != "fall_down"]

    se_hit = sum(1 for v in falls if v["fall_frames"] > 0)
    sp_ok = sum(1 for v in adls if v["fall_frames"] == 0)

    se = se_hit / len(falls) if falls else 0.0
    sp = sp_ok / len(adls) if adls else 0.0

    print()
    print(f" 【事件级口径】{title}")
    print(f"   跌倒敏感度 SE（视频中报出过跌倒） : {se_hit}/{len(falls)} = {se * 100:.1f}%")
    print(f"   跌倒特异度 SP（非跌倒视频零误报） : {sp_ok}/{len(adls)} = {sp * 100:.1f}%")
    if len(adls) - sp_ok > 0:
        bad = [k for k, v in data.items() if v["gt"] != "fall_down" and v["fall_frames"] > 0]
        print(f"   误报视频: {', '.join(bad[:8])}")
    return {"se": se, "sp": sp, "fall_n": len(falls), "adl_n": len(adls)}


def build_matrix(pairs):
    matrix = OrderedDict((gt, OrderedDict((p, 0) for p in CLASSES)) for gt in CLASSES)
    unknown = []
    for src, gt, pred in pairs:
        if gt not in matrix:
            continue
        if pred in matrix[gt]:
            matrix[gt][pred] += 1
        else:
            unknown.append((src, gt, pred))
    return matrix, unknown


def prf(matrix, cls):
    tp = matrix[cls][cls]
    fp = sum(matrix[gt][cls] for gt in CLASSES if gt != cls)
    fn = sum(matrix[cls][p] for p in CLASSES if p != cls)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1


def print_report(title, pairs):
    matrix, unknown = build_matrix(pairs)
    total = sum(sum(row.values()) for row in matrix.values())

    print()
    print("=" * 78)
    print(f" {title}")
    print("=" * 78)
    print(f" 视频总数: {total}" + (f"（另有 {len(unknown)} 个无法归类）" if unknown else ""))
    print()

    # ---- 混淆矩阵 ----
    print(" 混淆矩阵（行=真值，列=预测）")
    header = " " * 12 + "".join(f"{c[:8]:>10}" for c in CLASSES)
    print(header)
    for gt in CLASSES:
        row = "".join(
            f"{matrix[gt][p]:>10}" if matrix[gt][p] else f"{'.':>10}" for p in CLASSES
        )
        n = sum(matrix[gt].values())
        acc = matrix[gt][gt] / n if n else 0.0
        print(f" {gt:<11}{row}   acc={acc * 100:5.1f}%  (n={n})")

    print()
    # ---- 每类指标 ----
    print(" 分类指标")
    print(f" {'类别':<12}{'precision':>11}{'recall':>10}{'F1':>9}{'TP':>6}{'FP':>6}{'FN':>6}")
    for cls in CLASSES:
        tp, fp, fn, p, r, f1 = prf(matrix, cls)
        print(f" {cls:<12}{p * 100:>10.1f}%{r * 100:>9.1f}%{f1:>9.3f}{tp:>6}{fp:>6}{fn:>6}")

    # ---- 跌倒专项 SE / SP（综述要求的核心指标）----
    print()
    print(" 跌倒专项（Igual et al. 综述口径）")
    fall_tp = matrix["fall_down"]["fall_down"]
    fall_fn = sum(matrix["fall_down"][p] for p in CLASSES if p != "fall_down")
    fall_n = fall_tp + fall_fn
    se = fall_tp / fall_n if fall_n else 0.0

    fp_total = 0
    adl_total = 0
    fp_detail = []
    for gt in CLASSES:
        if gt == "fall_down":
            continue
        n = sum(matrix[gt].values())
        fp = matrix[gt]["fall_down"]
        adl_total += n
        fp_total += fp
        if fp:
            fp_detail.append(f"{gt}:{fp}/{n}")

    sp = 1.0 - (fp_total / adl_total) if adl_total else 0.0
    print(f"   敏感度 SE（真实跌倒→报跌倒）  : {se * 100:5.1f}%  ({fall_tp}/{fall_n})")
    print(f"   特异度 SP（非跌倒→不报跌倒）  : {sp * 100:5.1f}%  "
          f"(误报 {fp_total}/{adl_total})")
    if fp_detail:
        print(f"   误报来源: {', '.join(fp_detail)}")

    # ---- 总体精度 ----
    correct = sum(matrix[c][c] for c in CLASSES)
    print()
    print(f" 总体精度: {correct}/{total} = {correct / total * 100:.1f}%"
          if total else " 总体精度: N/A")

    # ---- 问题视频 ----
    problems = []
    for src, gt, pred in pairs:
        if gt != pred:
            problems.append((gt, pred, src))
    if problems:
        print()
        print(f" 误判视频明细（{len(problems)} 个）")
        for gt, pred, src in problems[:25]:
            print(f"   {src:<28} 真值={gt:<11} 预测={pred}")
        if len(problems) > 25:
            print(f"   ... 另有 {len(problems) - 25} 个")
    print("=" * 78)

    return {
        "total": total,
        "acc": correct / total if total else 0.0,
        "fall_se": se,
        "fall_sp": sp,
        "fall_fp": fp_total,
        "fall_fn": fall_fn,
    }


def main():
    ap = argparse.ArgumentParser(description="行为识别评估报告（含跌倒 SE/SP）")
    ap.add_argument("--csv", nargs="+", required=True, help="一份或多份评估 CSV")
    args = ap.parse_args()

    results = []
    event_results = []
    for path in args.csv:
        if not os.path.isfile(path):
            print(f"[SKIP] 找不到: {path}")
            continue
        pairs = read_summaries(path)
        title = os.path.basename(path)
        results.append((title, print_report(title, pairs)))
        # 事件级口径（跌倒检测的真实价值指标）
        ev = print_event_level(title, path)
        if ev:
            event_results.append((title, ev))

    # ---- 多份对比 ----
    if len(results) > 1:
        print()
        print("=" * 78)
        print(" 对比汇总")
        print("=" * 78)
        print(f" {'报告':<30}{'总体精度':>10}{'投票SE':>9}{'投票SP':>9}{'事件SE':>9}{'事件SP':>9}")
        ev_map = dict(event_results)
        for title, r in results:
            ev = ev_map.get(title, {})
            print(f" {title:<30}{r['acc'] * 100:>9.1f}%{r['fall_se'] * 100:>8.1f}%"
                  f"{r['fall_sp'] * 100:>8.1f}%"
                  f"{ev.get('se', 0) * 100:>8.1f}%{ev.get('sp', 0) * 100:>8.1f}%")
        print("=" * 78)
        print(" 注：投票口径=整段视频最多票的行为；事件级=视频中是否报出过跌倒（")
        print("     安防系统实际关心后者：跌倒发生在任意时刻都应报警）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
