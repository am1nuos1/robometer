cd /home/czhang30/robometer

# Defaults from scripts/batch_inference_local.py:
#   --model-path: required, no default
#   --input-dir: default None
#   --video: default []
#   --pattern: default "*.mp4"
#   --task: default None
#   --task-file: default None
#   --out-dir: default None, which means sibling "output/" next to input dir
#   --fps: default 1.0
#   --max-frames: default 8
#   --resize-max-side: default 224; use 0 to disable resizing
#   --debug-memory: default false
#   --success-threshold: default 0.5

.venv/bin/python scripts/batch_inference_local.py \
  --model-path robometer/Robometer-4B \
  --input-dir /home/czhang30/robometer/my_dataset/Krisha_dataset/head/observation.images.head \
  --pattern "*.mp4" \
  --task-file /home/czhang30/robometer/my_dataset/test1/task_description.txt \
  --out-dir /home/czhang30/robometer/my_dataset/test1/output \
  --fps 3.0 \
  --max-frames 30 \
  --resize-max-side 224 \
  --success-threshold 0.5
