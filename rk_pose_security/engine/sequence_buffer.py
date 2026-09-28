# ============================================================
# engine/sequence_buffer.py
#
# RK3588 Behavior Sequence Buffer
#
# Match training:
#
# Dataset:
#   seq_len=16
#   frame_stride=2
#
# Input:
#   feature (46,)
#
# Output:
#   (16,46)
#
# ============================================================


from collections import deque

import numpy as np



class PoseSequenceBuffer:



    def __init__(

        self,

        max_len=16,

        stride=2

    ):


        self.max_len=max_len


        # 与训练一致

        self.stride=stride



        # 每个人一个buffer

        self.buffers={}



        # 每个人采样计数

        self.counter={}



        print(

            "[Buffer] init",

            "len=",

            max_len,

            "stride=",

            stride

        )




    # =====================================================
    # update
    # =====================================================


    def update(

        self,

        track_id,

        feature

    ):


        try:


            if feature is None:

                return None




            # ==========================
            # init person
            # ==========================


            if track_id not in self.buffers:


                self.buffers[track_id]=deque(

                    maxlen=self.max_len

                )


                self.counter[track_id]=0




            # ==========================
            # stride sampling
            # ==========================


            self.counter[track_id]+=1



            if self.counter[track_id] % self.stride !=0:


                return None




            # ==========================
            # append feature
            # ==========================


            self.buffers[track_id].append(

                feature

            )




            # ==========================
            # wait 16 frames
            # ==========================


            if len(self.buffers[track_id]) < self.max_len:


                return None





            seq=np.asarray(

                self.buffers[track_id],

                dtype=np.float32

            )



            return seq




        except Exception as e:


            print(

                "[Buffer ERROR]",

                e

            )


            return None




    # =====================================================
    # clear
    # =====================================================


    def clear(

        self,

        track_id=None

    ):


        if track_id is None:


            self.buffers.clear()

            self.counter.clear()



        else:


            self.buffers.pop(

                track_id,

                None

            )


            self.counter.pop(

                track_id,

                None

            )




    # =====================================================
    # debug
    # =====================================================


    def length(

        self,

        track_id

    ):


        if track_id not in self.buffers:

            return 0


        return len(

            self.buffers[track_id]

        )
