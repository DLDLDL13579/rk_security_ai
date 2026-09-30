#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
摄像头 MJPEG 中转服务（供浏览器直接观看实时画面）

为什么需要它：
    摄像头 192.168.1.64 接在板端 eth2 专属网段（192.168.1.0/24），该网段不对外路由，
    Mac / 平台均无法直连。本服务在板端把 RTSP 转成浏览器可直接播放的 MJPEG 流，
    经板端 wlan0（192.168.31.123）对外提供，从而不触碰平台上任何项目。

为什么不用 `ffmpeg -f mpjpeg -listen 1`：
    实测其 HTTP 响应头为 `Content-Type: application/octet-stream`，
    浏览器会当作文件下载而非视频流；且该模式仅支持单连接，
    客户端断开即退出（导致 systemd 反复重启）。故改为本实现。

设计要点：
    ① 采集与分发分离：单个采集线程持续读取最新帧，多个客户端各自取帧，
       避免每个浏览器连接都去抢摄像头（海康设备并发拉流能力有限）
    ② 标准 MJPEG 响应头：multipart/x-mixed-replace; boundary=frame
    ③ 相机断流自动重连（指数退避），不影响已连接的客户端
    ④ 依赖仅 cv2 + 标准库；运行于 rk3588_clean 环境

环境变量：
    CAM_RTSP      摄像头 RTSP 地址（默认见下）
    CAM_PORT      监听端口（默认 8081）
    CAM_PATH      URL 路径（默认 /cam.mjpg）
    CAM_WIDTH     输出宽度（默认 1280，0=原始）
    CAM_FPS       目标帧率（默认 15）
    CAM_QUALITY   JPEG 质量 1-100（默认 70）
    CAM_CHANNEL   取流通道：main(101) / sub(102)，默认 sub 子码流

用法：
    python3 mjpeg_server.py
    浏览器打开 http://<板端IP>:8081/cam.mjpg
"""

import io
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# === 凭据处理（2026-09-30 安全整改）===
# 原实现把摄像头账号密码明文硬编码在此，并随提交进入了公开仓库（属凭据泄露）。
# 现改为从环境变量拼装，仓库内不再保存任何真实凭据：
#   方式一：CAM_RTSP 直接给完整地址（优先级最高，兼容旧用法）
#   方式二：CAM_USER / CAM_PASSWORD / CAM_HOST 三者齐备时自动拼装
# 板端由 systemd 单元的 Environment= 注入，不进版本库。
CAM_USER = os.environ.get("CAM_USER", "").strip()
CAM_PASSWORD = os.environ.get("CAM_PASSWORD", "").strip()
CAM_HOST = os.environ.get("CAM_HOST", "").strip()


def _mask_rtsp(url):
    """
    脱敏 RTSP 地址中的密码，用于日志与 /status 输出。

    原实现直接把含密码的完整地址打印出来（/status 是 HTTP 接口，
    浏览器与任何能访问 8081 的人都看得到），属凭据泄露。
    rtsp://user:pw@host:554/... → rtsp://user:***@host:554/...
    """
    if not url:
        return url
    try:
        head, tail = url.split("://", 1)
        if "@" not in tail:
            return url
        cred, hostpart = tail.rsplit("@", 1)
        if ":" not in cred:
            return url
        user = cred.split(":", 1)[0]
        return f"{head}://{user}:***@{hostpart}"
    except Exception:
        return "<rtsp>"


def _build_rtsp(channel):
    """按环境变量拼装 RTSP 地址；缺少凭据时返回空串，由下方统一报错"""
    if not (CAM_USER and CAM_PASSWORD and CAM_HOST):
        return ""
    return (
        f"rtsp://{CAM_USER}:{CAM_PASSWORD}@{CAM_HOST}:554"
        f"/Streaming/Channels/{channel}"
    )


DEFAULT_RTSP_MAIN = _build_rtsp("101")
DEFAULT_RTSP_SUB = _build_rtsp("102")

RTSP_URL = os.environ.get("CAM_RTSP", "").strip()
if not RTSP_URL:
    CHANNEL = os.environ.get("CAM_CHANNEL", "sub").strip().lower()
    RTSP_URL = DEFAULT_RTSP_MAIN if CHANNEL in ("main", "101") else DEFAULT_RTSP_SUB
if not RTSP_URL:
    raise SystemExit(
        "[ERROR] 未提供摄像头地址，且环境变量中没有可用凭据。请任选其一：\n"
        "        ① CAM_RTSP=rtsp://<user>:<password>@<ip>:554/Streaming/Channels/102\n"
        "        ② CAM_USER=<user> CAM_PASSWORD=<password> CAM_HOST=<ip>\n"
        "        （凭据不进版本库；板端见 rk-cam-mjpeg.service 的 Environment=）"
    )

PORT = int(os.environ.get("CAM_PORT", "8081"))
PATH = os.environ.get("CAM_PATH", "/cam.mjpg")
WIDTH = int(os.environ.get("CAM_WIDTH", "1280"))
TARGET_FPS = float(os.environ.get("CAM_FPS", "15"))
QUALITY = int(os.environ.get("CAM_QUALITY", "70"))
JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), QUALITY]
BOUNDARY = "frame"

BOUNDARY_LINE = ("--" + BOUNDARY + "\r\n").encode("ascii")
HEADER_TEMPLATE = (
    "--" + BOUNDARY + "\r\n"
    "Content-Type: image/jpeg\r\n"
    "Content-Length: {length}\r\n\r\n"
).encode("ascii")


class FrameHub:
    """单采集线程 + 多订阅者（最新帧覆盖）"""

    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self._jpeg = None
        self._seq = 0
        self._clients = 0
        self.running = True
        self.capture = None
        self.connected = False
        self.last_error = ""
        self.frames_read = 0
        self.reconnects = 0
        self.dropped_frames = 0
        self.frames_skipped = 0      # 因超出目标帧率而丢弃的帧数
        self._last_publish = 0.0
        # === ANNOTATED_CONSUMER_PATCH (2026-09-30) ===
        # 优先消费识别服务显示线程写出的【带标注】帧，让浏览器看到识别结果；
        # 该文件不存在或过期时自动回退为自行拉 RTSP 转发裸画面。
        self.annotated_path = os.environ.get(
            "CAM_ANNOTATED_JPEG", "/dev/shm/rk_annotated.jpg"
        ).strip()
        self.annotated_max_age = float(
            os.environ.get("CAM_ANNOTATED_MAX_AGE", "3.0")
        )
        self.annotated_active = False
        self.annotated_hits = 0
        # 读共享帧时的转码参数（带宽适配）
        self.annotated_resize = os.environ.get(
            "CAM_ANNOTATED_RESIZE", "1"
        ).strip().lower() not in ("0", "false", "no", "off")
        self.annotated_width = int(os.environ.get("CAM_WIDTH", "960"))
        self.annotated_quality = int(os.environ.get("CAM_QUALITY", "60"))

    # ------------------------------------------------------------------
    def _open(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("cannot open RTSP source")
        self.capture = cap

    def run(self):
        backoff = 1.0
        while self.running:
            # === ANNOTATED_CONSUMER_PATCH (2026-09-30) ===
            # 标注帧可用时：完全不碰摄像头，只转发识别结果
            if self._try_annotated():
                continue

            if self.capture is None:
                try:
                    self._open()
                    self.connected = True
                    self.last_error = ""
                    backoff = 1.0
                    print(f"[MJPEG] camera connected: {self.source}")
                except Exception as exc:
                    self.connected = False
                    self.last_error = str(exc)
                    self.reconnects += 1
                    print(f"[MJPEG] connect failed ({exc}), retry in {backoff:.0f}s")
                    self._sleep(backoff)
                    backoff = min(backoff * 2.0, 30.0)
                    continue

            if self._clients == 0:
                # 无观众时降低读取频率，省 CPU（但仍保持连接，避免反复重连）
                self._sleep(0.5)
                self._drain()

            ok, frame = self.capture.read()
            if not ok or frame is None:
                print("[MJPEG] read failed, reconnecting ...")
                self._release()
                self.connected = False
                self.reconnects += 1
                self._sleep(1.0)
                continue

            self.frames_read += 1

            # === 关键修复：始终全速读取，只按目标帧率「发布」 ===
            # 摄像头原生约 28fps。若在读帧之间 sleep(1/TARGET_FPS)，
            # 则每秒积压 (28 - TARGET_FPS) 帧，缓冲区越来越旧，
            # 表现为画面「慢放」且延迟持续增长（实测确认）。
            # 正确做法：全速 read() 把管道排空、多余帧直接丢弃，
            # 仅在距上次发布 >= 1/TARGET_FPS 时才编码并推送。
            now = time.time()
            if TARGET_FPS > 0 and (now - self._last_publish) < (1.0 / TARGET_FPS):
                self.frames_skipped += 1
                continue
            self._last_publish = now

            if WIDTH > 0 and frame.shape[1] != WIDTH:
                h = int(round(frame.shape[0] * WIDTH / float(frame.shape[1])))
                frame = cv2.resize(frame, (WIDTH, h))

            ok, buf = cv2.imencode(".jpg", frame, JPEG_PARAMS)
            if not ok:
                continue

            with self.lock:
                self._jpeg = buf.tobytes()
                self._seq += 1


    # === ANNOTATED_CONSUMER_PATCH (2026-09-30) ===
    def _try_annotated(self):
        """
        尝试消费识别服务产出的标注帧。

        返回 True 表示本轮已产出帧（调用方 continue）；
        False 表示标注帧不可用（不存在/过期/读取失败），调用方回退 RTSP。
        """
        if not self.annotated_path:
            return False

        try:
            st = os.stat(self.annotated_path)
            age = time.time() - st.st_mtime
            if age > self.annotated_max_age:
                if self.annotated_active:
                    print("[MJPEG] 标注帧过期，回退 RTSP 直连模式")
                    self.annotated_active = False
                if self.capture is not None:
                    self._release()
                    self.connected = False
                return False

            now = time.time()
            if TARGET_FPS > 0 and (now - self._last_publish) < (1.0 / TARGET_FPS):
                self.frames_skipped += 1
                return True
            self._last_publish = now

            with open(self.annotated_path, "rb") as fh:
                jpeg = fh.read()
            if len(jpeg) < 128:
                return True

            # === 带宽适配（2026-09-30 卡顿修复）===
            # 共享帧是板端显示分辨率（1920x1080，约 160KB/帧）。
            # 若直接转发，12fps 需要 1.9MB/s，而弱 WiFi 实测仅约 200KB/s
            # → 严重拥塞，浏览器只有 1~2 fps。此处按 CAM_WIDTH/CAM_QUALITY
            # 重新编码，把带宽压到 WiFi 可承受范围。
            if self.annotated_resize and (
                self.annotated_width > 0 or self.annotated_quality != 75
            ):
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    if (
                        self.annotated_width > 0
                        and img.shape[1] != self.annotated_width
                    ):
                        h = int(
                            round(
                                img.shape[0]
                                * self.annotated_width
                                / float(img.shape[1])
                            )
                        )
                        img = cv2.resize(img, (self.annotated_width, h))
                    ok, buf = cv2.imencode(
                        ".jpg",
                        img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.annotated_quality],
                    )
                    if ok:
                        jpeg = buf.tobytes()

            with self.lock:
                self._jpeg = jpeg
                self._seq += 1
            self.frames_read += 1
            self.annotated_hits += 1

            if not self.annotated_active:
                print("[MJPEG] 已切换到【标注帧】模式（浏览器将看到识别结果）")
                self.annotated_active = True
            return True
        except FileNotFoundError:
            if self.annotated_active:
                print("[MJPEG] 标注帧消失，回退 RTSP 直连模式")
                self.annotated_active = False
            return False
        except Exception as exc:
            print(f"[MJPEG] 标注帧读取失败（回退 RTSP）：{exc}")
            self.annotated_active = False
            return False

    def _drain(self):
        """丢弃管道中积压的旧帧，避免恢复观看时播放延迟画面"""
        if self.capture is None:
            return
        for _ in range(3):
            self.capture.grab()

    def _sleep(self, seconds):
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(0.2, max(0.0, end - time.time())))

    def _release(self):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
            self.capture = None

    # ------------------------------------------------------------------
    def latest(self, last_seq):
        with self.lock:
            if self._seq == last_seq:
                return None, last_seq
            return self._jpeg, self._seq

    def add_client(self):
        with self.lock:
            self._clients += 1

    def remove_client(self):
        with self.lock:
            self._clients = max(0, self._clients - 1)

    def status(self):
        with self.lock:
            return {
                "camera_connected": self.connected,
                "frames_read": self.frames_read,
                "clients": self._clients,
                "reconnects": self.reconnects,
                "dropped_frames": self.dropped_frames,
                "frames_skipped": self.frames_skipped,
                # ANNOTATED_CONSUMER_PATCH: 当前画面来源（annotated=带识别标注）
                "mode": "annotated" if self.annotated_active else "rtsp_raw",
                "annotated_path": self.annotated_path,
                "last_error": self.last_error,
                # 脱敏：/status 是 HTTP 接口，不能回显含密码的完整 RTSP 地址
                "source": _mask_rtsp(self.source),
            }

    def stop(self):
        self.running = False
        self._release()


HUB = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "rk-mjpeg/1.0"

    def log_message(self, fmt, *args):  # 静默常规访问日志
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._send_index()
            return

        if path == PATH:
            self._send_stream()
            return

        if path == "/status":
            import json

            body = json.dumps(HUB.status(), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "not found")

    # ------------------------------------------------------------------
    def _send_index(self):
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>RK3588 实时画面</title>
<style>
 body{{margin:0;background:#111;color:#eee;font-family:system-ui,-apple-system,sans-serif}}
 header{{padding:10px 16px;background:#1b1b1b;font-size:14px}}
 img{{display:block;width:100%;height:auto;background:#000}}
 .meta{{padding:8px 16px;font-size:13px;color:#9aa}}
 code{{color:#7fd}}
</style></head>
<body>
<header>RK3588 安防 · 实时画面（摄像头子码流 MJPEG 中转）</header>
<img src="{PATH}" alt="live">
<div class="meta">地址：<code>{PATH}</code> ｜ 状态：<code><a href="/status" style="color:#7fd">/status</a></code></div>
</body></html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_stream(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        HUB.add_client()
        last_seq = -1
        try:
            while True:
                jpeg, last_seq = HUB.latest(last_seq)
                if jpeg is None:
                    time.sleep(0.01)
                    continue

                # 实时性保护：若内核发送队列已积压，说明客户端消费不过来。
                # 此时丢弃当前帧而不是继续排队，避免延迟累积（实测曾积压 270KB
                # 导致画面延迟数分钟）。实时画面只保证「最新」，不保证「完整」。
                if not self._write_ready():
                    HUB.dropped_frames += 1
                    time.sleep(0.02)
                    continue

                self.wfile.write(BOUNDARY_LINE)
                self.wfile.write(HEADER_TEMPLATE.replace(b"{length}", str(len(jpeg)).encode()))
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            print(f"[MJPEG] client error: {exc}")
        finally:
            HUB.remove_client()

    def _write_ready(self, max_pending=32768):
        """
        检查本连接的发送队列是否通畅。

        TCP_INFO 的 tcpi_notsent_bytes + 未确认字节数过大即视为积压；
        取不到信息时保守返回 True（退化为原行为）。
        """
        try:
            info = self.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 24)
            if len(info) >= 24:
                # struct tcp_info: 前 4 字节 state，随后为各 u32 计数
                # unacked 在偏移 16，snd_nxt 在偏移 20；这里用 unacked 近似积压
                unacked = int.from_bytes(info[16:20], "little")
                return unacked <= max_pending
        except Exception:
            pass
        return True


def main():
    global HUB

    print("=" * 62)
    print(" RK3588 Camera MJPEG relay")
    print("=" * 62)
    print(f" source   : {_mask_rtsp(RTSP_URL)}")
    print(f" listen   : http://0.0.0.0:{PORT}{PATH}")
    print(f" width    : {WIDTH}  fps: {TARGET_FPS}  quality: {QUALITY}")
    print("=" * 62)

    HUB = FrameHub(RTSP_URL)
    threading.Thread(target=HUB.run, name="capture", daemon=True).start()

    # 等待首帧，避免浏览器一开始连进来看到空白
    waited = 0.0
    while waited < 30.0:
        jpeg, _ = HUB.latest(-1)
        if jpeg is not None:
            print("[MJPEG] first frame ready")
            break
        time.sleep(0.3)
        waited += 0.3
    else:
        print("[MJPEG] warning: no frame within 30s, still serving")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MJPEG] stopping ...")
    finally:
        HUB.stop()
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
