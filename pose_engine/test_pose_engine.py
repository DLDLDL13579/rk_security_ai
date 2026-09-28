from engine import PoseEngine

engine = PoseEngine()

results = engine.run(
    "../output/pose_0005.json"
)

for r in results:
    print(r)
