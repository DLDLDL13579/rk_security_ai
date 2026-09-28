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

        h0,w0 = img.shape[:2]

        img = cv2.resize(img,(640,640))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = np.expand_dims(img,0)

        return img,w0,h0

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

    def postprocess(self,outputs,orig_w,orig_h):

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

        sx=orig_w/640
        sy=orig_h/640

        results=[]

        for i in keep:

            x1,y1,x2,y2=boxes[i]

            results.append({

                "bbox":[
                    int(x1*sx),
                    int(y1*sy),
                    int(x2*sx),
                    int(y2*sy)
                ],

                "score":float(scores[i]),
                "class":int(class_ids[i]),
                "name":CLASSES[class_ids[i]]

            })

        return results

    def detect(self,img):

        inp,w,h=self.preprocess(img)

        outputs=self.rknn.inference(inputs=[inp])

        outputs=[np.array(x,dtype=np.float32) for x in outputs]

        return self.postprocess(outputs,w,h)
