#!/bin/bash
echo "=================================="
echo "Pose Engine RTSP Realtime Detect"
echo "=================================="

    python /home/nvidia/test/PaddleDetection/deploy/python/det_keypoint_unite_infer.py \
    --det_model_dir=/home/nvidia/test/PaddleDetection/output_inference/ppyolo_r50vd_dcn_1x_coco \
    --keypoint_model_dir=/home/nvidia/test/PaddleDetection/models/tinypose_256x192 \
    --camera_id=0 \
    --device=GPU