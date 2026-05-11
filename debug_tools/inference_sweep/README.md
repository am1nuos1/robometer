# Inference Sweep

This folder is for local inference debugging sweeps. Edit `sweep_config.json`, then run `run_sweep.py`.

Example:

```bash
cd /home/czhang30/robometer

.venv/bin/python debug_tools/inference_sweep/run_sweep.py --dry-run

.venv/bin/python debug_tools/inference_sweep/run_sweep.py --clean
```

Parameter sweep examples:

```json
"quantization": [true, false]
```

Write all values manually:

```json
"resize_max_side": [224, 336, 448]
```

Equivalent explicit form:

```json
"resize_max_side": {"values": [224, 336, 448]}
```

Range-style form without an end value. It uses `start`, `step`, and `count`:

```json
"max_frames": {"start": 4, "step": 4, "count": 4}
```

This expands to:

```json
[4, 8, 12, 16]
```

If you do not want to write `count` for every parameter, set top-level `range_count` once:

```json
"range_count": 3,
"sweep": {
  "fps": {"start": 1.0, "step": 0.5}
}
```

This expands to:

```json
[1.0, 1.5, 2.0]
```

Known labels can be used for filtering configs later:

```json
"expected_success": {
  "ep_000": true,
  "ep_001": false,
  "ep_002": true,
  "ep_003": false
}
```

Each run gets its own output directory under `output_root`, plus a `.log` file. The summary is written to:

```text
debug_tools/inference_sweep/runs/summary.csv
debug_tools/inference_sweep/runs/summary.jsonl
```

Useful summary columns:

```text
pairwise_label_accuracy
success_failure_margin
ep_000_reward_mean
ep_001_reward_mean
ep_002_reward_mean
ep_003_reward_mean
```
