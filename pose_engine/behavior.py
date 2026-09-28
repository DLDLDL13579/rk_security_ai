def classify_behavior(features):

    left_elbow = features["left_elbow"]
    right_elbow = features["right_elbow"]

    left_knee = features["left_knee"]
    right_knee = features["right_knee"]

    body_tilt = features["body_tilt"]

    # 跌倒
    if body_tilt < 30:
        return "fall_down"

    # 下蹲
    if left_knee < 120 and right_knee < 120:
        return "squat"

    # 弯腰
    if 30 <= body_tilt <= 70:
        return "bend"

    # 举左手
    if left_elbow < 80 and right_elbow > 120:
        return "raise_left_hand"

    # 举右手
    if right_elbow < 80 and left_elbow > 120:
        return "raise_right_hand"

    # 双手举起
    if left_elbow < 80 and right_elbow < 80:
        return "raise_both_hands"

    return "standing"
