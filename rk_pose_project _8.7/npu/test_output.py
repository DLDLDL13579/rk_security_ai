import cv2
import numpy as np
from rknnlite.api import RKNNLite

rknn = RKNNLite()

rknn.load_rknn("model.rknn")
rknn.init_runtime()

img = cv2.imread("test.jpg")

img = cv2.resize(img,(640,640))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img = np.expand_dims(img,0)

outputs = rknn.inference(inputs=[img])

for i,out in enumerate(outputs):

    out=np.array(out)

    print("\n===================")
    print("output",i)
    print("shape =",out.shape)
    print("min =",out.min())
    print("max =",out.max())
    print("mean =",out.mean())

    print(out.flatten()[:20])
