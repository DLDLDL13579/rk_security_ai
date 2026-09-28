# Purpose: Verify PoseSequenceBuffer accepts 46D feature vectors and emits a (16, 46) sequence.
# This is a lightweight unit test and does not require RKNN models.

import numpy as np

from engine.sequence_buffer import PoseSequenceBuffer


print("=" * 60)
print("Sequence Buffer Basic Test")
print("=" * 60)

buffer = PoseSequenceBuffer(max_len=16)
person_id = 0

for frame_id in range(40):
    feature = np.array([
        0.5,
        179.6,
        0.62,
        *np.zeros(43, dtype=np.float32),
    ], dtype=np.float32)

    seq = buffer.update(person_id, feature)

    print(
        "frame:", frame_id,
        "buffer:", buffer.length(person_id),
        "output:", None if seq is None else seq.shape,
    )

    if seq is not None:
        break

print("TEST DONE")
