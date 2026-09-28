import os

cmd = """
python deploy/python/det_keypoint_unite_infer.py 
--det_model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco 
--keypoint_model_dir=models/tinypose_256x192 
--camera_id=0 
--device=GPU
"""

os.system(cmd)
