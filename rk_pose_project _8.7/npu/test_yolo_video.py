import cv2

from yolo_rknn_v2 import YOLO_RKNN

model = YOLO_RKNN("model.rknn")

cap = cv2.VideoCapture(r"/home/admin/PaddleDetection/rk_pose_project/npu/input_video/test.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.detect(frame)

    for det in results:

        if det["name"] != "person":
            continue

        x1,y1,x2,y2 = det["bbox"]

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f'{det["name"]} {det["score"]:.2f}',
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255),
            2
        )

    cv2.imshow("YOLO RK3588", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
