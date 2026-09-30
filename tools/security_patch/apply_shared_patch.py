#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SharedData 增量补丁 —— 为权威工程 engine/shared.py 补齐安防字段

设计原则（零回归）：
    ① 只在原 SharedData 类中**新增**字段与方法，绝不修改任何已有方法
    ② 幂等：重复执行不会重复插入
    ③ 自动备份原件（.bak_security_<时间戳>）
    ④ 插入后做语法校验（py_compile）

为什么需要这个补丁：
    权威工程的 SharedData 缺 events / security_states / security_detections /
    zone_overlays 四个字段（_8.4 版本有），显示端与安全线程需要它们。
    SecurityThread 已用 hasattr 探测做了容错，故本补丁是**可选增强**，
    不执行也能跑，只是显示端无法绘制区域框与事件。

用法（板端）：
    python3 apply_shared_patch.py /path/to/rk_pose_authoritative/engine/shared.py
    python3 apply_shared_patch.py --check /path/to/.../shared.py   # 只检查是否已打过
"""

import argparse
import py_compile
import shutil
import sys
import time
from pathlib import Path

MARKER = "# === SECURITY PATCH (security_monitor) ==="

FIELDS_SNIPPET = """        self._events = []
        self._security_states = {}
        self._security_detections = []
        self._zone_overlays = []
"""

METHODS_SNIPPET = '''
    # === SECURITY PATCH (security_monitor) ===
    # 以下方法由 security_patch 增量追加，用于安防事件/区域状态共享。
    # 原有方法未被改动。

    def set_events(self, events):
        with self.lock:
            self._events = copy.deepcopy(events)

    def get_events(self):
        with self.lock:
            return copy.deepcopy(self._events)

    def set_security_states(self, security_states):
        with self.lock:
            self._security_states = copy.deepcopy(security_states)

    def get_security_states(self):
        with self.lock:
            return copy.deepcopy(self._security_states)

    def set_security_detections(self, detections):
        with self.lock:
            self._security_detections = copy.deepcopy(detections)

    def get_security_detections(self):
        with self.lock:
            return copy.deepcopy(self._security_detections)

    def set_zone_overlays(self, zones):
        with self.lock:
            self._zone_overlays = copy.deepcopy(zones)

    def get_zone_overlays(self):
        with self.lock:
            return copy.deepcopy(self._zone_overlays)
'''


def already_patched(text):
    return MARKER in text


def apply_patch(path):
    p = Path(path)
    if not p.is_file():
        print(f"[ERROR] 文件不存在: {p}")
        return 2

    text = p.read_text(encoding="utf-8")

    if already_patched(text):
        print(f"[SKIP] 已打过补丁，无需重复执行: {p}")
        return 0

    # ---- 校验目标结构 ----
    if "class SharedData" not in text:
        print("[ERROR] 未找到 class SharedData，文件可能不是权威工程的 shared.py")
        return 3
    if "self.lock = threading.Lock()" not in text:
        print("[ERROR] 未找到 self.lock 初始化行，无法安全插入字段")
        return 3
    if "import copy" not in text:
        print("[ERROR] 缺少 import copy（补丁依赖 deepcopy）")
        return 3

    # ---- 1) 在 __init__ 的 self.lock 之后插入字段 ----
    anchor = "        self.lock = threading.Lock()\n"
    if FIELDS_SNIPPET in text:
        text_fields = text
    else:
        text_fields = text.replace(anchor, anchor + FIELDS_SNIPPET, 1)

    if text_fields == text:
        print("[ERROR] 字段插入失败（锚点未命中）")
        return 4

    # ---- 2) 在类尾追加方法（追加到文件末尾，缩进 4 空格，仍在类作用域内
    #         的前提是文件末尾就是类体结束；为稳妥起见，插到最后一个类的
    #         末尾之后不成立，故改为插到文件末尾并显式缩进）
    # 说明：Python 中类体到文件结束仍在类内，只要中间没有顶格代码。
    # 因此这里直接追加到文件末尾即等价于加到 SharedData 类里。
    patched = text_fields.rstrip("\n") + "\n" + METHODS_SNIPPET

    # ---- 备份 + 写入 ----
    backup = p.with_suffix(p.suffix + f".bak_security_{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(p, backup)
    p.write_text(patched, encoding="utf-8")

    # ---- 语法校验 ----
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(backup, p)
        print(f"[ERROR] 语法校验失败，已自动回滚: {exc}")
        return 5

    print(f"[OK] 已打补丁: {p}")
    print(f"[OK] 备份: {backup}")
    print("[OK] 语法校验通过")
    return 0


def main():
    ap = argparse.ArgumentParser(description="SharedData 安防字段增量补丁")
    ap.add_argument("path", help="shared.py 路径")
    ap.add_argument("--check", action="store_true", help="只检查是否已打补丁")
    args = ap.parse_args()

    if args.check:
        p = Path(args.path)
        if not p.is_file():
            print(f"[ERROR] 文件不存在: {p}")
            return 2
        patched = already_patched(p.read_text(encoding="utf-8"))
        print("[YES] 已打补丁" if patched else "[NO] 未打补丁")
        return 0 if patched else 1

    return apply_patch(args.path)


if __name__ == "__main__":
    sys.exit(main())
