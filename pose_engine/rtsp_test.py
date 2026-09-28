import os
import cv2

rtsp_url = os.environ.get("RK_RTSP_URL", "")

cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("RTSP打开失败")
    exit()

print("RTSP连接成功")

while True:

    ret, frame = cap.read()

    if not ret:
        print("读取失败")
        break

    cv2.imshow("RTSP", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
