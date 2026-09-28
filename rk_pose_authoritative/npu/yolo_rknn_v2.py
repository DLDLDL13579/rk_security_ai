import cv2
import numpy as np
from rknnlite.api import RKNNLite


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

ANCHORS = [
    [(10,13),(16,30),(33,23)],
    [(30,61),(62,45),(59,119)],
    [(116,90),(156,198),(373,326)]
]

CLASSES = [
"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
"traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
"dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
"umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
"bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
"sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
"couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
"remote","keyboard","cell phone","microwave","oven","toaster","sink",
"refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
"toothbrush"
]


class YOLO_RKNN:

    def __init__(self, model_path):

        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError("load rknn failed")

        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("init runtime failed")

        print("[YOLO] RKNN loaded")

        self.input_size = 640

    def preprocess(self, img):

        h0, w0 = img.shape[:2]

        # letterbox：保持宽高比，不足处用灰色(114)补齐，避免人物被拉伸变形
        scale = min(
            self.input_size / float(w0),
            self.input_size / float(h0),
        )
        nw = max(1, int(round(w0 * scale)))
        nh = max(1, int(round(h0 * scale)))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full(
            (self.input_size, self.input_size, 3),
            114,
            dtype=np.uint8,
        )
        pad_x = (self.input_size - nw) // 2
        pad_y = (self.input_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        inp = np.expand_dims(canvas, 0)
        return inp, scale, pad_x, pad_y

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def nms(self,boxes,scores):

        x1=boxes[:,0]
        y1=boxes[:,1]
        x2=boxes[:,2]
        y2=boxes[:,3]

        areas=(x2-x1+1)*(y2-y1+1)

        order=scores.argsort()[::-1]

        keep=[]

        while order.size>0:

            i=order[0]
            keep.append(i)

            xx1=np.maximum(x1[i],x1[order[1:]])
            yy1=np.maximum(y1[i],y1[order[1:]])
            xx2=np.minimum(x2[i],x2[order[1:]])
            yy2=np.minimum(y2[i],y2[order[1:]])

            w=np.maximum(0,xx2-xx1+1)
            h=np.maximum(0,yy2-yy1+1)

            inter=w*h

            ovr=inter/(areas[i]+areas[order[1:]]-inter)

            inds=np.where(ovr<=NMS_THRESH)[0]

            order=order[inds+1]

        return keep

    def decode_layer(self,pred,anchors,stride):

        pred=pred[0]

        c,h,w=pred.shape

        pred=pred.reshape(3,85,h,w)

        boxes=[]
        scores=[]
        class_ids=[]

        for a in range(3):

            anchor_w,anchor_h=anchors[a]

            feat=pred[a]

            for gy in range(h):
                for gx in range(w):

                    det=feat[:,gy,gx]

                    obj=det[4]

                    if obj<OBJ_THRESH:
                        continue

                    cls=np.argmax(det[5:])
                    cls_score=det[5+cls]

                    score=obj*cls_score

                    if score<OBJ_THRESH:
                        continue

                    x=(det[0]*2-0.5+gx)*stride
                    y=(det[1]*2-0.5+gy)*stride

                    bw=(det[2]*2)**2*anchor_w
                    bh=(det[3]*2)**2*anchor_h

                    x1=x-bw/2
                    y1=y-bh/2
                    x2=x+bw/2
                    y2=y+bh/2

                    boxes.append([x1,y1,x2,y2])
                    scores.append(score)
                    class_ids.append(cls)

        return boxes,scores,class_ids

    def postprocess(self, outputs, scale, pad_x, pad_y):

        boxes=[]
        scores=[]
        class_ids=[]

        strides=[8,16,32]

        for out,anchor,stride in zip(outputs,ANCHORS,strides):

            b,s,c=self.decode_layer(
                out,
                anchor,
                stride
            )

            boxes.extend(b)
            scores.extend(s)
            class_ids.extend(c)

        if len(boxes)==0:
            return []

        boxes=np.array(boxes)
        scores=np.array(scores)

        keep=self.nms(boxes,scores)

        inv = 1.0 / scale
        ow = int(round((self.input_size - 2 * pad_x) * inv))
        oh = int(round((self.input_size - 2 * pad_y) * inv))

        results=[]

        for i in keep:

            x1,y1,x2,y2=boxes[i]

            x1 = int(round((x1 - pad_x) * inv))
            y1 = int(round((y1 - pad_y) * inv))
            x2 = int(round((x2 - pad_x) * inv))
            y2 = int(round((y2 - pad_y) * inv))
            x1 = max(0, min(x1, ow))
            y1 = max(0, min(y1, oh))
            x2 = max(0, min(x2, ow))
            y2 = max(0, min(y2, oh))

            results.append({

                "bbox":[
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "score":float(scores[i]),
                "class":int(class_ids[i]),
                "name":CLASSES[class_ids[i]]

            })

        return results

    def detect(self,img):

        inp, scale, pad_x, pad_y = self.preprocess(img)

        outputs=self.rknn.inference(inputs=[inp])

        outputs=[np.array(x,dtype=np.float32) for x in outputs]

        return self.postprocess(outputs, scale, pad_x, pad_y)
