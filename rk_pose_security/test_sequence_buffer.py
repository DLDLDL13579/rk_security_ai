# Purpose: Verify PoseSequenceBuffer stride behavior and final sequence shape using fake 46D features.
# This is a lightweight unit test and does not require RKNN models.

import numpy as np

from engine.sequence_buffer import PoseSequenceBuffer


print("=" * 60)
print("Sequence Buffer Shape Test")
print("=" * 60)

buffer = PoseSequenceBuffer(max_len=16)
person_id = 1

for i in range(40):
    fake_feature = np.random.rand(46).astype(np.float32)
    seq = buffer.update(person_id, fake_feature)

    print("Frame:", i + 1, "Buffer:", buffer.length(person_id))

    if seq is not None:
        print()
        print("Sequence Ready!")
        print("Shape:", seq.shape)
        print("dtype:", seq.dtype)
        print("first frame:", seq[0][:10])
        break

print()
print("TEST DONE")
