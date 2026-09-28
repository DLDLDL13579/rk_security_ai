# ============================================================
# npu/pose_rknn.py
#
# RK3588 Smart Security AI Platform
#
# YOLOv5-Pose RKNN Wrapper
#
# Version:
#     Pose Decode V1.1
#
# Date:
#     2026-07-31
#
# Pipeline:
#
#     YOLOv5-Pose RKNN
#              |
#              |
#          RAW Output
#              |
#              |
#        Pose Decode
#              |
#              |
#        17 Keypoints
#              |
#              |
#        MediaPipe 33 Format
#
#
# Interface:
#
#     detect(img)
#
# Return:
#
#     list[33]
#
# Compatible:
#
#     FeatureExtractorV3
#
# ============================================================


import cv2
import numpy as np


from rknnlite.api import RKNNLite




# ============================================================
# Landmark
# ============================================================


class Landmark:


    def __init__(
            self,
            x,
            y,
            z=0.0,
            visibility=1.0
    ):

        self.x=float(x)

        self.y=float(y)

        self.z=float(z)

        self.visibility=float(visibility)






# ============================================================
# Pose RKNN
# ============================================================


class PoseRKNN:


    def __init__(

            self,

            model_path

    ):


        print("[Pose] Loading...")


        self.rknn=RKNNLite()



        ret=self.rknn.load_rknn(

            model_path

        )


        if ret!=0:

            raise RuntimeError(
                "load pose rknn failed"
            )



        ret=self.rknn.init_runtime()



        if ret!=0:

            raise RuntimeError(
                "init pose runtime failed"
            )



        self.input_size=640


        self.conf_threshold=0.35



        print("[Pose] OK")





    # ========================================================
    # preprocess
    # ========================================================


    def preprocess(
            self,
            img
    ):


        img=cv2.resize(

            img,

            (
                self.input_size,
                self.input_size
            )

        )


        img=cv2.cvtColor(

            img,

            cv2.COLOR_BGR2RGB

        )


        img=np.expand_dims(

            img,

            0

        )


        return img





    # ========================================================
    # sigmoid
    # ========================================================


    def sigmoid(
            self,
            x
    ):

        return 1/(1+np.exp(-x))





    # ========================================================
    # decode
    # ========================================================


    def decode(

            self,

            outputs,

            w,

            h

    ):


        if outputs is None:

            return None



        # 当前先取最大目标

        best_score=0

        best_kpts=None



        strides=[8,16,32]



        for output,stride in zip(

                outputs,

                strides

        ):


            output=np.squeeze(

                output,

                0

            )


            C,H,W=output.shape



            output=np.transpose(

                output,

                (1,2,0)

            )



            for yy in range(H):

                for xx in range(W):


                    data=output[yy,xx]



                    obj=self.sigmoid(

                        data[4]

                    )


                    if obj < self.conf_threshold:

                        continue



                    kpts=[]


                    start=85



                    for i in range(17):


                        px=data[start+i*3]

                        py=data[start+i*3+1]

                        pv=data[start+i*3+2]



                        px=self.sigmoid(px)

                        py=self.sigmoid(py)

                        pv=self.sigmoid(pv)



                        kpts.append(

                            Landmark(

                                px,

                                py,

                                0,

                                pv

                            )

                        )



                    if obj>best_score:


                        best_score=obj


                        best_kpts=kpts





        if best_kpts is None:

            return None




        return self.convert17to33(

            best_kpts

        )





    # ========================================================
    # COCO17 -> MediaPipe33
    # ========================================================


    def convert17to33(

            self,

            pts

    ):


        result=[

            Landmark(

                0,

                0,

                0,

                1.0

            )

            for _ in range(33)

        ]



        mapping={


            0:0,


            1:2,

            2:5,


            3:7,

            4:8,


            5:11,

            6:12,


            7:13,

            8:14,


            9:15,

            10:16,


            11:23,

            12:24,


            13:25,

            14:26,


            15:27,

            16:28

        }




        for src,dst in mapping.items():


            result[dst]=pts[src]





        # 填充缺失点

        for i in range(33):


            if result[i].visibility <=0:


                result[i]=Landmark(

                    result[i-1].x,

                    result[i-1].y,

                    0,

                    1.0

                )



        return result





    # ========================================================
    # detect
    # ========================================================


    def detect(

            self,

            img

    ):


        if img is None:

            return None



        h,w=img.shape[:2]



        inp=self.preprocess(

            img

        )



        outputs=self.rknn.inference(

            inputs=[inp]

        )



        if outputs is None:

            return None




        if not hasattr(

                self,

                "_debug"

        ):


            self._debug=True


            print("================")

            print("POSE OUTPUT")


            for i,o in enumerate(outputs):

                print(

                    i,

                    o.shape

                )


            print("================")





        kpts=self.decode(

            outputs,

            w,

            h

        )



        if kpts is None:

            return None




        print(

            "[POSE]",

            len(kpts)

        )



        return kpts





    # ========================================================
    # release
    # ========================================================


    def release(self):

        self.rknn.release()
