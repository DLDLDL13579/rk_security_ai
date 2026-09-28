import math


class PoseEngine:

    def __init__(self):

        self.behaviors = [
            "站立",
            "下蹲",
            "跌倒",
            "弯腰",
            "举左手",
            "举右手",
            "双手举起"
        ]

    # =====================================
    # 三点角度
    # =====================================

    def calc_angle(self, a, b, c):

        ba = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]

        dot = ba[0] * bc[0] + ba[1] * bc[1]

        norm1 = math.sqrt(
            ba[0] ** 2 + ba[1] ** 2
        )

        norm2 = math.sqrt(
            bc[0] ** 2 + bc[1] ** 2
        )

        if norm1 == 0 or norm2 == 0:
            return 0

        cos_angle = dot / (norm1 * norm2)

        cos_angle = max(-1, min(1, cos_angle))

        return round(
            math.degrees(
                math.acos(cos_angle)
            ),
            2
        )

    # =====================================
    # 身体倾斜角
    # =====================================

    def body_tilt(self, kp):

        ls = kp[5]
        rs = kp[6]

        lh = kp[11]
        rh = kp[12]

        sx = (ls[0] + rs[0]) / 2
        sy = (ls[1] + rs[1]) / 2

        hx = (lh[0] + rh[0]) / 2
        hy = (lh[1] + rh[1]) / 2

        dx = hx - sx
        dy = hy - sy

        angle = abs(
            math.degrees(
                math.atan2(dy, dx)
            )
        )

        return round(angle, 2)

    # =====================================
    # 特征提取
    # =====================================

    def extract_feature(self, person):

        kp = person["keypoints"]

        left_elbow = self.calc_angle(
            kp[5],
            kp[7],
            kp[9]
        )

        right_elbow = self.calc_angle(
            kp[6],
            kp[8],
            kp[10]
        )

        left_knee = self.calc_angle(
            kp[11],
            kp[13],
            kp[15]
        )

        right_knee = self.calc_angle(
            kp[12],
            kp[14],
            kp[16]
        )

        tilt = self.body_tilt(kp)

        return {

            "person_id":
                person["person_id"],

            "left_elbow":
                left_elbow,

            "right_elbow":
                right_elbow,

            "left_knee":
                left_knee,

            "right_knee":
                right_knee,

            "body_tilt":
                tilt
        }

    # =====================================
    # 行为识别
    # =====================================

    def classify_behavior(self, feature):

        left_elbow = feature["left_elbow"]
        right_elbow = feature["right_elbow"]

        left_knee = feature["left_knee"]
        right_knee = feature["right_knee"]

        body_tilt = feature["body_tilt"]

        if body_tilt < 30:
            return "跌倒"

        if left_knee < 120 and right_knee < 120:
            return "下蹲"

        if 30 <= body_tilt <= 70:
            return "弯腰"

        if left_elbow < 80 and right_elbow > 120:
            return "举左手"

        if right_elbow < 80 and left_elbow > 120:
            return "举右手"

        if left_elbow < 80 and right_elbow < 80:
            return "双手举起"

        return "站立"

    # =====================================
    # 单人分析
    # =====================================

    def analyze_person(self, person):

        feature = self.extract_feature(
            person
        )

        feature["behavior"] = \
            self.classify_behavior(
                feature
            )

        return feature

    # =====================================
    # 多人分析
    # =====================================

    def analyze(self, pose_data):

        results = []

        for person in pose_data:

            result = self.analyze_person(
                person
            )

            results.append(result)

        return results
