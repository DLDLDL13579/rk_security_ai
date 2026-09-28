import os
import fastdeploy as fd
import cv2

MODEL_DIR = "/home/nvidia/test/PaddleDetection/mode_test(lw)/bottle/picodet_s_416_coco_lcnet"
RTSP_URL = os.environ.get("RK_RTSP_URL", "")
CONF_THRESH = 0.05  # 降低阈值

option = fd.RuntimeOption()
option.use_cpu()
model = fd.vision.detection.PicoDet(
    model_file=f"{MODEL_DIR}/model.pdmodel",
    params_file=f"{MODEL_DIR}/model.pdiparams",
    config_file=f"{MODEL_DIR}/infer_cfg.yml",
    runtime_option=option
)

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    result = model.predict(frame)
    boxes = result.boxes
    scores = result.scores
    label_ids = result.label_ids

    bottle_count = 0
    for box, score, label_id in zip(boxes, scores, label_ids):
        if label_id == 44 and score >= CONF_THRESH:
            bottle_count += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"bottle: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f"瓶子坐标: ({x1},{y1})-({x2},{y2}), 置信度: {score:.3f}")

    # 缩放显示
    display = cv2.resize(frame, (960, 540))
    cv2.imshow("Bottle Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()