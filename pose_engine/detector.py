import json


def load_pose_json(json_path):

    with open(json_path, "r") as f:
        data = json.load(f)

    persons = []

    # =====================================
    # 格式1：
    # pose_0005.json
    # =====================================
    if isinstance(data[0], dict):

        return data

    # =====================================
    # 格式2：
    # det_keypoint_unite_image_results.json
    # =====================================
    image_info = data[0]

    bbox_list = image_info[1]

    keypoints_all = image_info[2][0]

    for person_id, keypoints in enumerate(keypoints_all):

        persons.append({

            "person_id": person_id,

            "bbox": bbox_list[person_id],

            "keypoints": keypoints

        })

    return persons
