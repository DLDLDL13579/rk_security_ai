#!/bin/bash
# ============================================================
# 一键测试 data/videos/ 全部视频（按子目录名作为标签）
# 输出: output/eval_all_<时间戳>.csv
# ============================================================
set -e
cd "$(dirname "$0")"

# ---- 自动探测可用的 Python 解释器（板端无 `python` 命令，见 find_python.sh）----
source "$(dirname "$0")/find_python.sh"
mkdir -p output

TS=$(date +%Y%m%d_%H%M)
OUT="output/eval_all_${TS}.csv"

echo "=== Eval ALL videos -> $OUT ==="
"$RK_PYTHON_BIN" eval/test_multi_behavior_eval.py --source data/videos --out "$OUT"

echo "RESULT: $OUT"
