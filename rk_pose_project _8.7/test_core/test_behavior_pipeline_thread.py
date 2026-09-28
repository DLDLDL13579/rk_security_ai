# =====================================================
# test_behavior_pipeline_thread.py
#
# Optimized Behavior Demo
#
# MediaPipe Pose
# + Feature
# + TCN
# + Smooth Skeleton
#
# =====================================================


import os
import sys
import cv2
import time
import threading
import numpy as np



# =====================================================
# ROOT
# =====================================================


ROOT=os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.insert(
    0,
    ROOT
)



from pose.mediapipe_pose import MediaPipePose

from engine.feature import PoseFeatureExtractor

from engine.sequence_buffer import PoseSequenceBuffer

from npu.behavior_rknn import BehaviorRKNN




VIDEO=ROOT+"/videos/standing/stand_007.mp4"


MODEL=ROOT+"/npu/onnx-rknn/behavior_tcn.rknn"




# =====================================================
# shared
# =====================================================


class Shared:


    def __init__(self):

        self.lock=threading.Lock()

        self.frame=None

        self.result={}

        self.running=True



shared=Shared()




# =====================================================
# skeleton
# =====================================================


CONNECTIONS=[

(11,12),

(11,13),
(13,15),

(12,14),
(14,16),

(15,17),
(17,19),
(19,21),

(16,18),
(18,20),
(20,22),


(11,23),
(23,25),
(25,27),
(27,29),
(29,31),


(12,24),
(24,26),
(26,28),
(28,30),
(30,32)

]





# =====================================================
# smooth
# =====================================================


class PoseSmooth:


    def __init__(self):

        self.last=None

        self.alpha=0.65



    def update(self,kpts):


        arr=np.array(
            [
                [
                p.x,
                p.y,
                p.visibility
                ]
                for p in kpts
            ],
            dtype=np.float32
        )



        if self.last is None:

            self.last=arr

        else:

            self.last=(

                self.alpha*self.last

                +

                (1-self.alpha)*arr

            )



        return self.last







# =====================================================
# draw skeleton
# =====================================================


def draw_skeleton(

        img,

        pts

):


    h,w=img.shape[:2]



    xy=[]



    for p in pts:


        x=int(
            p[0]*w
        )

        y=int(
            p[1]*h
        )

        xy.append(
            (x,y)
        )


        if p[2]>0.3:


            cv2.circle(

                img,

                (x,y),

                3,

                (0,0,255),

                -1

            )



    for a,b in CONNECTIONS:


        if a>=len(xy) or b>=len(xy):

            continue



        cv2.line(

            img,

            xy[a],

            xy[b],

            (255,200,0),

            1,

            cv2.LINE_AA

        )





# =====================================================
# bbox
# =====================================================


def bbox_from_pose(

        pts,

        w,

        h

):


    xs=[]

    ys=[]


    for p in pts:


        if p[2]>0.3:

            xs.append(
                int(p[0]*w)
            )

            ys.append(
                int(p[1]*h)
            )



    if len(xs)<5:

        return None



    pad=20


    return (

        max(0,min(xs)-pad),

        max(0,min(ys)-pad),

        min(w,max(xs)+pad),

        min(h,max(ys)+pad)

    )





# =====================================================
# video
# =====================================================


def video_worker():


    cap=cv2.VideoCapture(
        VIDEO
    )


    while shared.running:


        ret,frame=cap.read()


        if not ret:

            cap.release()

            cap=cv2.VideoCapture(
                VIDEO
            )

            continue



        frame=cv2.resize(

            frame,

            (640,360)

        )


        with shared.lock:


            shared.frame=frame



        time.sleep(
            0.03
        )





# =====================================================
# AI
# =====================================================


def ai_worker():


    pose=MediaPipePose()


    feature=PoseFeatureExtractor()


    buffer=PoseSequenceBuffer(
        16
    )


    model=BehaviorRKNN(
        MODEL
    )



    smooth=PoseSmooth()



    last=0


    while shared.running:



        with shared.lock:

            frame=shared.frame



        if frame is None:

            continue



        now=time.time()



        # 8 FPS AI

        if now-last<0.12:

            time.sleep(0.01)

            continue



        last=now




        kpts=pose.detect(
            frame
        )


        if kpts is None:

            continue




        smooth_pts=smooth.update(
            kpts
        )



        feat=feature.extract(
            kpts
        )


        if feat is None:

            continue




        seq=buffer.update(

            0,

            feat

        )



        action=""

        score=0




        if seq is not None:


            action,score=model.forward(
                seq
            )


            print(

                "[ACTION]",

                action,

                score

            )





        bbox=bbox_from_pose(

            smooth_pts,

            frame.shape[1],

            frame.shape[0]

        )



        with shared.lock:


            shared.result={


                "pose":

                    smooth_pts,


                "bbox":

                    bbox,


                "action":

                    action,


                "score":

                    score


            }








# =====================================================
# display
# =====================================================


def display_worker():


    last=time.time()

    fps=0



    while shared.running:



        with shared.lock:

            frame=shared.frame


            result=shared.result.copy()



        if frame is None:

            continue



        img=frame.copy()



        if "pose" in result:


            draw_skeleton(

                img,

                result["pose"]

            )



        if result.get("bbox"):


            x1,y1,x2,y2=result["bbox"]



            cv2.rectangle(

                img,

                (x1,y1),

                (x2,y2),

                (0,255,0),

                2

            )


            text=(

            "person | "

            +

            result["action"]

            +

            " %.2f"

            %

            result["score"]

            )



            cv2.putText(

                img,

                text,

                (x1,y1-8),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0,255,255),

                2

            )




        now=time.time()



        if now-last>1:


            fps=1/(now-last)

            last=now




        cv2.putText(

            img,

            "FPS %.1f"%fps,

            (20,30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255,0,0),

            2

        )



        cv2.imshow(

            "Pose Behavior Demo",

            cv2.resize(

                img,

                (960,540)

            )

        )



        if cv2.waitKey(1)&0xff==27:

            shared.running=False




    cv2.destroyAllWindows()





# =====================================================
# main
# =====================================================


print(
"START"
)


threads=[


threading.Thread(
target=video_worker,
daemon=True
),


threading.Thread(
target=ai_worker,
daemon=True
),


threading.Thread(
target=display_worker,
daemon=True
)

]



for t in threads:

    t.start()



try:

    while shared.running:

        time.sleep(1)


except KeyboardInterrupt:

    shared.running=False



print(
"EXIT"
)
