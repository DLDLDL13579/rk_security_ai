#!/bin/bash
# ============================================================
# find_python.sh —— 自动探测可用的 Python 解释器
#
# 背景（2026-09-28 实测）：
#   板端**没有 `python` 命令**（只有 `python3`），
#   而系统 python3（3.10）**没有任何项目依赖**（缺 rknnlite/cv2/mediapipe）。
#   原脚本用裸 `python`，导致 `bash run_demo.sh` 直接失败：
#       run_demo.sh: line 44: exec: python: not found
#
#   依赖完整的环境只有 conda 的 rk3588_clean（实测 5/5 依赖齐全）。
#
# 本脚本被 run_demo.sh / run_eval_all.sh 引用（source）。
# 探测顺序：
#   ① $RK_PYTHON 环境变量（用户显式指定，最高优先级）
#   ② 已知 conda 环境（rk3588_clean 优先，因其依赖最全）
#   ③ conda base
#   ④ 系统 python3（最后兜底，可能缺依赖）
#
# 探测标准：该解释器能同时 import rknnlite 与 cv2。
#
# 用法：
#   source "$(dirname "$0")/find_python.sh"
#   "$RK_PYTHON_BIN" app/main_realtime.py
# ============================================================

_fp_candidates() {
  # ① 用户显式指定
  if [ -n "$RK_PYTHON" ]; then
    echo "$RK_PYTHON"
  fi

  # ② 已知 conda 环境（rk3588_clean 依赖最全，优先）
  for _env in rk3588_clean rk3588 rknn test onnx_deploy; do
    for _root in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
      [ -x "$_root/envs/$_env/bin/python" ] && echo "$_root/envs/$_env/bin/python"
    done
  done

  # ③ conda base
  for _root in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -x "$_root/bin/python" ] && echo "$_root/bin/python"
  done

  # ④ 系统兜底
  command -v python3 2>/dev/null
  command -v python 2>/dev/null
}

_fp_has_deps() {
  "$1" -c "import rknnlite, cv2" >/dev/null 2>&1
}

RK_PYTHON_BIN=""
RK_PYTHON_OK=0

while IFS= read -r _cand; do
  [ -n "$_cand" ] || continue
  [ -x "$_cand" ] || continue

  if _fp_has_deps "$_cand"; then
    RK_PYTHON_BIN="$_cand"
    RK_PYTHON_OK=1
    break
  fi

  [ -z "$RK_PYTHON_BIN" ] && RK_PYTHON_BIN="$_cand"
done <<EOF
$(_fp_candidates)
EOF

if [ -z "$RK_PYTHON_BIN" ]; then
  echo "[错误] 未找到任何 Python 解释器" >&2
  echo "       请设置 RK_PYTHON 指向可用解释器，例如：" >&2
  echo "       export RK_PYTHON=\$HOME/miniconda3/envs/rk3588_clean/bin/python" >&2
  exit 1
fi

export RK_PYTHON_BIN

if [ "$RK_PYTHON_OK" != "1" ]; then
  echo "[警告] 所选解释器缺少必需依赖（rknnlite / cv2）：" >&2
  echo "       $RK_PYTHON_BIN" >&2
  echo "       建议：pip install -r requirements.txt" >&2
  echo "       或指定：export RK_PYTHON=<path>/bin/python" >&2
fi
