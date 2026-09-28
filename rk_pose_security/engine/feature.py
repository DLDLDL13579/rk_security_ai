# ============================================================
# engine/feature.py
#
# RK3588 Behavior Feature Extractor V5
#
# Compatible:
#   MediaPipe Pose
#   PoseRKNN
#
# Output:
#   46D feature
#
# Same as training
#
# ============================================================


import numpy as np



class PoseFeatureExtractor:



    def __init__(self):


        self.prev_joints = {}


        self.eps = 1e-6


        print(
            "[Feature] FeatureExtractorV5 ready"
        )




    # ------------------------------------------------
    # vector
    # ------------------------------------------------

    def vec(self,a,b):

        ax, ay = self.xy(a)
        bx, by = self.xy(b)

        return np.array(
            [
                ax-bx,
                ay-by
            ],
            dtype=np.float32
        )

    def xy(self, point):

        if hasattr(point, "x") and hasattr(point, "y"):

            return float(point.x), float(point.y)

        arr = np.asarray(point, dtype=np.float32).reshape(-1)

        if arr.size < 2:

            raise ValueError("invalid point shape")

        return float(arr[0]), float(arr[1])

    def visibility(self, point):

        if hasattr(point, "visibility"):

            return float(point.visibility)

        arr = np.asarray(point, dtype=np.float32).reshape(-1)

        if arr.size >= 4:

            return float(arr[3])

        if arr.size >= 3:

            return float(arr[2])

        return 0.0



    # ------------------------------------------------
    # angle
    # ------------------------------------------------

    def angle(self,a,b,c):

        ba=self.vec(
            a,
            b
        )


        bc=self.vec(
            c,
            b
        )


        denom=(

            np.linalg.norm(ba)
            *
            np.linalg.norm(bc)

            +
            self.eps

        )


        cos=np.dot(
            ba,
            bc
        ) / denom



        return float(

            np.arccos(

                np.clip(
                    cos,
                    -1,
                    1
                )

            )

        )




    # ------------------------------------------------
    # extract
    # ------------------------------------------------


    def extract(

            self,

            kpts,

            track_id=0

    ):



        if kpts is None:

            return None



        if len(kpts)<33:

            return None



        # COCO / MediaPipe index


        nose=kpts[0]


        ls=kpts[11]

        rs=kpts[12]


        lh=kpts[23]

        rh=kpts[24]


        lk=kpts[25]

        rk=kpts[26]


        la=kpts[27]

        ra=kpts[28]



        points=[

            nose,
            ls,
            rs,
            lh,
            rh,
            lk,
            rk,
            la,
            ra

        ]


        if any(
            p is None
            for p in points
        ):

            return None




        # ==================================================
        # 1 Spatial 18D
        #
        # 注意:
        # 保持0~1坐标体系
        #
        # 不恢复bbox
        #
        # ==================================================


        lhx, lhy = self.xy(lh)
        rhx, rhy = self.xy(rh)
        lsx, lsy = self.xy(ls)
        rsx, rsy = self.xy(rs)

        cx=(lhx+rhx)/2

        cy=(lhy+rhy)/2



        scale=(

            np.sqrt(

                (lsx-rsx)**2
                +
                (lsy-rsy)**2

            )

            +
            self.eps

        )



        def norm(p):

            px, py = self.xy(p)

            return [

                (px-cx)/scale,

                (py-cy)/scale

            ]



        spatial=np.array(

            norm(ls)
            +
            norm(rs)
            +
            norm(lh)
            +
            norm(rh)
            +
            norm(lk)
            +
            norm(rk)
            +
            norm(la)
            +
            norm(ra)
            +
            norm(nose),

            dtype=np.float32

        )



        # ==================================================
        # 2 Angle 8D
        # ==================================================


        angles=np.array(

            [

            self.angle(
                lh,
                lk,
                la
            ),


            self.angle(
                rh,
                rk,
                ra
            ),


            self.angle(
                ls,
                lh,
                lk
            ),


            self.angle(
                rs,
                rh,
                rk
            ),


            self.angle(
                lh,
                ls,
                rs
            ),


            self.angle(
                lk,
                lh,
                rh
            ),


            self.angle(
                nose,
                ls,
                lh
            ),


            self.angle(
                nose,
                rs,
                rh
            )

            ],

            dtype=np.float32

        )



        # ==================================================
        # 3 Symmetry 4D
        # ==================================================


        sym=np.array(

            [

            abs(
                self.xy(lk)[1]-self.xy(rk)[1]
            ),


            abs(
                lsy-rsy
            ),


            abs(
                lhy-rhy
            ),


            abs(
                self.xy(la)[1]-self.xy(ra)[1]
            )

            ],

            dtype=np.float32

        )



        # ==================================================
        # 4 Velocity 16D
        # ==================================================


        joints=np.array(

            [

            lsx,lsy,

            rsx,rsy,


            lhx,lhy,

            rhx,rhy,


            *self.xy(lk),

            *self.xy(rk),


            *self.xy(la),

            *self.xy(ra)

            ],

            dtype=np.float32

        )



        if track_id not in self.prev_joints:


            velocity=np.zeros_like(

                joints

            )


        else:


            velocity=(

                joints
                -
                self.prev_joints[track_id]

            )



        self.prev_joints[track_id]=joints.copy()



        # 防止异常速度

        velocity=np.clip(

            velocity,

            -0.05,

            0.05

        )



        # ==================================================
        # final 46D
        #
        # 18+8+4+16
        #
        # ==================================================


        feature=np.concatenate(

            [

            spatial,

            angles,

            sym,

            velocity

            ]

        )



        return feature.astype(

            np.float32

        )
