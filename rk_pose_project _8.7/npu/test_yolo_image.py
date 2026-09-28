# test_yolo_image.py
# 这个测试代码能完美运行
import cv2
from yolo_rknn_v2 import YOLO_RKNN

yolo = YOLO_RKNN("model.rknn")

img = cv2.imread(r"/home/admin/PaddleDetection/rk_pose_project/npu/input_img/test.jpg")

results = yolo.detect(img)

for r in results:

    print(r)

    x1,y1,x2,y2 = r["bbox"]

    cv2.rectangle(
        img,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    cv2.putText(
        img,
        r["name"],
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,0,255),
        2
    )

cv2.imwrite(r"/home/admin/PaddleDetection/rk_pose_project/npu/output-img/result.jpg",img)

print("saved result.jpg")
