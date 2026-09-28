#!/bin/bash
# ============================================================
# 一键测试 data/videos/ 全部视频（按子目录名作为标签）
# 输出: output/eval_all_<时间戳>.csv
# ============================================================
set -e
cd "$(dirname "$0")"
mkdir -p output

TS=$(date +%Y%m%d_%H%M)
OUT="output/eval_all_${TS}.csv"

echo "=== Eval ALL videos -> $OUT ==="
python eval/test_multi_behavior_eval.py --source data/videos --out "$OUT"

echo "RESULT: $OUT"
