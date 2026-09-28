# ONNX → RKNN 转换说明（行为 TCN）

## 环境要求（PC 上执行，板端不需要）

```bash
# 需要 rknn-toolkit2（PC 版，不是 rknnlite）
pip install rknn-toolkit2
# 验证脚本需要 onnxruntime（可选）
pip install onnxruntime
```

## 使用步骤

1. 在训练工程导出 ONNX：

   ```bash
   cd ~/PaddleDetection/train_test2_zhb_new
   python pipelines/export_onnx.py
   ```

   得到 `outputs/model/behavior_tcn.onnx`，拷贝到本目录（`onnx_rknn/`）。

2. 转换（推荐用真实训练数据做 INT8 量化校准）：

   ```bash
   python convert_behavior_tcn.py \
       --onnx behavior_tcn.onnx \
       --calib-dir /home/gkfd/train_test2_zhb_new/data/processed/dataset
   ```

   输出 `onnx_rknn/behavior_tcn.rknn`。

3. （可选）转换后对比 ONNX 与 RKNN 输出：

   ```bash
   python convert_behavior_tcn.py --verify
   ```

4. 部署到推理工程：

   ```bash
   cp behavior_tcn.rknn ../npu/onnx-rknn/behavior_tcn.rknn
   ```

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--no-quantize` | 不量化（float16），量化效果不好时试 |
| `--dataset calib.txt` | 自定义校准清单（每行一个 npy 绝对路径） |
| `--calib-per-class 30` | 调整每类校准样本数 |
| `--target-platform rk3588` | 目标平台 |

## 校准数据位置

校准数据就是训练用的 46D 特征样本（.npy），放在
`onnx_rknn/calib/<类别>/` 下，每类一个子目录：

```text
onnx_rknn/calib/
  standing/  walking/  squat/  bend/  fall_down/
```

工程已预置每类 20 个（共 100 个），转换时：

```bash
python convert_behavior_tcn.py \
    --onnx behavior_tcn.onnx \
    --calib-dir onnx_rknn/calib
```

若训练数据集在别的机器/目录，也可以直接指向它，无需复制：

```bash
python convert_behavior_tcn.py --calib-dir /home/gkfd/train_test2_zhb_new/data/processed/dataset
```

## 常见问题

- **构建报 dynamic shape / batch 相关错误**：导出 ONNX 时去掉 `dynamic_axes`
  或固定 batch=1。脚本已默认通过 `load_onnx(inputs=["input"],
  input_size_list=[[1,16,46]])` 把动态 batch 固定为 1，无需再改 ONNX。
- **报 Opset 19 不支持 / opset 超 15 错误**：脚本会自动把 opset 降级为 15
  再转换（生成 `behavior_tcn_opset15.onnx`）。根治方法是用训练工程的
  `pipelines/export_onnx.py`（opset 12）重新导出 onnx。
- **量化后精度下降明显**：先试 `--no-quantize` 确认是量化损失还是模型问题；
  再用更多真实校准样本（建议每类 20~50 个，覆盖 5 类分布）。
- **输出只有一类**：检查校准数据是否覆盖所有类别（脚本默认每类取 20 个）。
