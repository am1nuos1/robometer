cd /home/czhang30/robometer

.venv/bin/python scripts/batch_inference_local.py \
  --model-path robometer/Robometer-4B \
  --input-dir /home/czhang30/robometer/my_dataset/test1/input \
  --pattern "*.mp4" \
  --task-file /home/czhang30/robometer/my_dataset/test1/task_description.txt \
  --out-dir /home/czhang30/robometer/my_dataset/test1/output \
  --fps 1.0 \
  --max-frames 8 \
  --resize-max-side 224 \
  --success-threshold 0.5



.venv/bin/python scripts/batch_inference_local.py \
  --model-path robometer/Robometer-4B \
  --input-dir /home/czhang30/robometer/my_dataset/Krisha_dataset/head/observation.images.head \
  --pattern "*.mp4" \
  --task-file /home/czhang30/robometer/my_dataset/test1/task_description.txt \
  --fps 3.0 \
  --max-frames 30 \
  --resize-max-side 224 \
  --success-threshold 0.5


# letf arm
# /home/czhang30/robometer/my_dataset/Left_arm_picks_a_cube_places_in_box/task_description.txt
.venv/bin/python scripts/batch_inference_local.py \
  --model-path robometer/Robometer-4B \
  --input-dir /home/czhang30/robometer/my_dataset/Left_arm_picks_a_cube_places_in_box/head/observation.images.head \
  --task-file /home/czhang30/robometer/my_dataset/Left_arm_picks_a_cube_places_in_box/task_description.txt \
  --fps 3.0 \
  --max-frames 30 \
  --resize-max-side 224 \
  --success-threshold 0.5
/home/czhang30/robometer/my_dataset/Left_arm_picks_a_cube_places_in_box/left_wrist/observation.images.left_wrist