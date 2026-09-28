"""
====================================================
Sequence Buffer Test V1.0

Test:

46 dim feature
        |
        ↓
PoseSequenceBuffer
        |
        ↓
(16,46)

====================================================
"""


import numpy as np


from engine.sequence_buffer import PoseSequenceBuffer



print("="*60)
print(" Sequence Buffer Test ")
print("="*60)



buffer = PoseSequenceBuffer(
    max_len=16
)



person_id=1



# =========================
# fake feature
# 模拟feature.py输出
# =========================


for i in range(20):


    fake_feature={

        "vector":
        np.random.rand(
            46
        ).astype(
            np.float32
        )

    }



    seq = buffer.update(

        person_id,

        fake_feature

    )



    print(
        "Frame:",
        i+1,
        "Buffer:",
        len(
            buffer.buffers[person_id]
        )
    )



    if seq is not None:


        print(
            "\nSequence Ready!"
        )


        print(
            "Shape:",
            seq.shape
        )


        print(
            "dtype:",
            seq.dtype
        )


        print(
            "first frame:",
            seq[0][:10]
        )


        break



print("\nTEST DONE")
