from pose_engine.detector import load_pose_json
from pose_engine.feature import extract_features
from pose_engine.behavior import classify_behavior


class PoseEngine:

    def __init__(self):

        print("Pose Engine Init")

    # =========================
    # 实时模式
    # =========================
    def process_person(self, person):

        features = extract_features(person)

        behavior = classify_behavior(features)

        return {
            "person_id": person["person_id"],
            "behavior": behavior,
            "features": features
        }

    # =========================
    # 离线JSON模式
    # =========================
    def run(self, json_file):

        persons = load_pose_json(json_file)

        results = []

        for person in persons:

            features = extract_features(person)

            behavior = classify_behavior(features)

            results.append({

                "person_id": person["person_id"],

                "bbox": person["bbox"],

                "behavior": behavior,

                "features": features
            })

        return results
