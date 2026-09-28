from engine import PoseEngine

engine = PoseEngine()

results = engine.run(
    "../det_keypoint_unite_image_results.json"
)

for r in results:

    print(
        r["person_id"],
        r["behavior"]
    )
