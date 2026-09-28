import math

def calc_angle(a, b, c):

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot = ba[0] * bc[0] + ba[1] * bc[1]

    norm1 = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    norm2 = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if norm1 == 0 or norm2 == 0:
        return 0

    cos_angle = dot / (norm1 * norm2)
    cos_angle = max(-1, min(1, cos_angle))

    angle = math.degrees(math.acos(cos_angle))

    return round(angle, 2)


def body_tilt(keypoints):

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    sx = (left_shoulder[0] + right_shoulder[0]) / 2
    sy = (left_shoulder[1] + right_shoulder[1]) / 2

    hx = (left_hip[0] + right_hip[0]) / 2
    hy = (left_hip[1] + right_hip[1]) / 2

    dx = hx - sx
    dy = hy - sy

    angle = abs(math.degrees(math.atan2(dy, dx)))

    return round(angle, 2)


def extract_features(person):

    pid = person["person_id"]

    kp = person["keypoints"]

    left_elbow = calc_angle(kp[5], kp[7], kp[9])
    right_elbow = calc_angle(kp[6], kp[8], kp[10])

    left_knee = calc_angle(kp[11], kp[13], kp[15])
    right_knee = calc_angle(kp[12], kp[14], kp[16])

    tilt = body_tilt(kp)

    return {
        "person_id": pid,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "body_tilt": tilt
    }
