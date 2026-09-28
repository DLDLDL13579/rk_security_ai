# core/（预留抽象层）

当前版本的主流程（YOLO 检测结果 → `engine/object_engine.py` 的 Detection 对象）暂不需要额外抽象层，本目录预留用于未来扩展：

- 多目标跟踪器的统一接口（ByteTrack/DeepSort 接入点）
- 检测结果的跨模块 DTO 定义
- 多视频源（RTSP/本地/网络流）适配器

需要时可在本目录新增 `detection.py`、`adapter.py` 等模块，并保持与 `engine/` 的解耦。
