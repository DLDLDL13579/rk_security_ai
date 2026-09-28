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


import os
import numpy as np



class PoseFeatureExtractor:



    def __init__(self):


        self.prev_joints = {}


        self.eps = 1e-6
        self.velocity_deadzone = float(
            os.environ.get("RK_VELOCITY_DEADZONE", "0.0035")
        )


        print(
            "[Feature] FeatureExtractorV5 ready"
        )




    # ------------------------------------------------
    # vector
    # ------------------------------------------------

    def vec(self,a,b):

        return np.array(
            [
                a.x-b.x,
                a.y-b.y
            ],
            dtype=np.float32
        )



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


        cx=(lh.x+rh.x)/2

        cy=(lh.y+rh.y)/2



        scale=(

            np.sqrt(

                (ls.x-rs.x)**2
                +
                (ls.y-rs.y)**2

            )

            +
            self.eps

        )



        def norm(p):

            return [

                (p.x-cx)/scale,

                (p.y-cy)/scale

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
                lk.y-rk.y
            ),


            abs(
                ls.y-rs.y
            ),


            abs(
                lh.y-rh.y
            ),


            abs(
                la.y-ra.y
            )

            ],

            dtype=np.float32

        )



        # ==================================================
        # 4 Velocity 16D
        # ==================================================


        joints=np.array(

            [

            ls.x,ls.y,

            rs.x,rs.y,


            lh.x,lh.y,

            rh.x,rh.y,


            lk.x,lk.y,

            rk.x,rk.y,


            la.x,la.y,

            ra.x,ra.y

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

        if self.velocity_deadzone > 0.0:

            velocity[np.abs(velocity) < self.velocity_deadzone] = 0.0



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

    def clear_track(self, track_id):
        self.prev_joints.pop(track_id, None)
