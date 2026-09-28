persons = keypoint_res['keypoint'][0]

for pid, kp in enumerate(persons):

    person = {
        "person_id": pid,
        "keypoints": kp
    }

    try:

        result = pose_engine.process_person(person)

        behavior = result["behavior"]

        x = int(kp[0][0])
        y = int(kp[0][1])

        cv2.putText(
            im,
            behavior,
            (x, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    except:
        pass
fps_count += 1

if fps_count >= 30:

    now = time.time()

    fps = fps_count / (now - fps_time)

    print(
        "FPS:",
        round(fps,2)
    )

    fps_count = 0

    fps_time = now
ret, frame = capture.read()

for _ in range(3):
    capture.grab()
