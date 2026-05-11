#!/usr/bin/env python3
"""
Batch local inference for multiple videos that share the same task description.

Example:
  uv run python scripts/batch_inference_local.py \
    --model-path robometer/Robometer-4B \
    --input-dir my_videos \
    --task "Pick up the green cup" \
    --out-dir my_videos/output
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from robometer.data.dataset_types import ProgressSample, Trajectory
from robometer.evals.eval_server import compute_batch_outputs
from robometer.evals.eval_viz_utils import create_combined_progress_success_plot, extract_frames
from robometer.utils.save import load_model_from_hf
from robometer.utils.setup_utils import setup_batch_collator


def load_frames_input(
    video_or_array_path: str,
    *,
    fps: float = 1.0,
    max_frames: int = 512,
    resize_max_side: int = 224,
) -> np.ndarray:
    """Load frames from a video path/URL or .npy/.npz file. Returns uint8 (T, H, W, C)."""
    if video_or_array_path.endswith(".npy"):
        frames_array = np.load(video_or_array_path)
    elif video_or_array_path.endswith(".npz"):
        with np.load(video_or_array_path, allow_pickle=False) as npz:
            if "frames" in npz:
                frames_array = npz["frames"].copy()
            elif "arr_0" in npz:
                frames_array = npz["arr_0"].copy()
            else:
                frames_array = next(iter(npz.values())).copy()
    else:
        frames_array = extract_frames(video_or_array_path, fps=fps, max_frames=max_frames)
        if frames_array is None or frames_array.size == 0:
            raise RuntimeError(f"Could not extract frames from video: {video_or_array_path}")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    if frames_array.ndim == 4 and frames_array.shape[1] in (1, 3) and frames_array.shape[-1] not in (1, 3):
        frames_array = frames_array.transpose(0, 2, 3, 1)
    frames_array = resize_frames_max_side(frames_array, resize_max_side)
    return frames_array


def resize_frames_max_side(frames: np.ndarray, max_side: int) -> np.ndarray:
    """Resize HWC video frames so the longest side is at most max_side."""
    if max_side is None or max_side <= 0 or frames.ndim != 4:
        return frames

    height, width = frames.shape[1], frames.shape[2]
    longest = max(height, width)
    if longest <= max_side:
        return frames

    scale = max_side / float(longest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        resized.append(np.asarray(img, dtype=np.uint8))
    return np.stack(resized, axis=0)


def log_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[cuda] {label}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB")


def describe_batch_inputs(batch_inputs: dict) -> None:
    for key, value in batch_inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"[batch] {key}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
        elif isinstance(value, list):
            print(f"[batch] {key}: list len={len(value)}")


class LocalRewardRunner:
    """Load the model once and reuse it across multiple videos."""

    def __init__(self, model_path: str, device: Optional[torch.device] = None, debug_memory: bool = False):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.exp_config, self.tokenizer, self.processor, self.reward_model = load_model_from_hf(
            model_path=model_path,
            device=device,
        )
        self.reward_model.eval()
        self.debug_memory = bool(debug_memory)
        if self.debug_memory:
            log_cuda_memory("after model load")
        self.batch_collator = setup_batch_collator(self.processor, self.tokenizer, self.exp_config, is_eval=True)

        loss_config = getattr(self.exp_config, "loss", None)
        self.is_discrete = (
            getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
            if loss_config
            else False
        )
        self.num_bins = (
            getattr(loss_config, "progress_discrete_bins", None)
            or getattr(self.exp_config.model, "progress_discrete_bins", 10)
        )

    def compute_rewards_per_frame(
        self,
        video_frames: np.ndarray,
        task: str,
        sample_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        t_steps = int(video_frames.shape[0])
        traj = Trajectory(
            frames=video_frames,
            frames_shape=tuple(video_frames.shape),
            task=task,
            id=sample_id,
            metadata={"subsequence_length": t_steps},
            video_embeddings=None,
        )
        progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
        batch = self.batch_collator([progress_sample])
        if self.debug_memory:
            describe_batch_inputs(batch["progress_inputs"])

        progress_inputs = batch["progress_inputs"]
        for key, value in progress_inputs.items():
            if hasattr(value, "to"):
                progress_inputs[key] = value.to(self.device)
        if self.debug_memory:
            describe_batch_inputs(progress_inputs)
            log_cuda_memory("before forward")

        try:
            results = compute_batch_outputs(
                self.reward_model,
                self.tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=self.is_discrete,
                num_bins=self.num_bins,
            )
        except torch.cuda.OutOfMemoryError:
            log_cuda_memory("OOM during forward")
            raise
        if self.debug_memory:
            log_cuda_memory("after forward")

        progress_pred = results.get("progress_pred", [])
        progress_array = (
            np.array(progress_pred[0], dtype=np.float32)
            if progress_pred and len(progress_pred) > 0
            else np.array([], dtype=np.float32)
        )

        outputs_success = results.get("outputs_success", {})
        success_probs = outputs_success.get("success_probs", []) if outputs_success else []
        success_array = (
            np.array(success_probs[0], dtype=np.float32)
            if success_probs and len(success_probs) > 0
            else np.array([], dtype=np.float32)
        )

        return progress_array, success_array


def resolve_videos(input_dir: Optional[str], videos: list[str], pattern: str) -> list[Path]:
    resolved: list[Path] = []

    if input_dir:
        for match in sorted(Path(input_dir).glob(pattern)):
            if match.is_file():
                resolved.append(match)

    for video in videos:
        matches = glob.glob(video)
        if matches:
            resolved.extend(Path(m) for m in sorted(matches) if Path(m).is_file())
        else:
            path = Path(video)
            if path.is_file():
                resolved.append(path)

    deduped: list[Path] = []
    seen = set()
    for path in resolved:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)

    if not deduped:
        raise ValueError("No input videos found. Use --input-dir and/or --video.")

    return deduped


def read_task(task: Optional[str], task_file: Optional[str]) -> str:
    if task_file:
        text = Path(task_file).read_text(encoding="utf-8").strip()
        if text:
            return text
    if task:
        return task.strip()
    raise ValueError("Provide --task or --task-file.")


def resolve_output_dir(
    input_dir: Optional[str],
    videos: list[Path],
    out_dir: Optional[str],
) -> Path:
    if out_dir:
        return Path(out_dir)

    if input_dir:
        input_path = Path(input_dir)
        return input_path.parent / "output"

    first_video_parent = videos[0].parent
    return first_video_parent.parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch local inference for multiple videos that share the same task.",
    )
    parser.add_argument("--model-path", required=True, help="HuggingFace model id or local checkpoint path")
    parser.add_argument("--input-dir", default=None, help="Directory containing input videos")
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Video path or glob pattern; can be passed multiple times",
    )
    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern used with --input-dir (default: *.mp4)",
    )
    parser.add_argument("--task", default=None, help="Shared task instruction for all videos")
    parser.add_argument("--task-file", default=None, help="Text file containing the shared task instruction")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for output .npy/.png files (default: sibling 'output/' next to input dir)",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="FPS when sampling from video (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=8, help="Max frames to extract from video (default: 8)")
    parser.add_argument(
        "--resize-max-side",
        type=int,
        default=224,
        help="Resize frames so the longest side is at most this many pixels; use 0 to disable (default: 224)",
    )
    parser.add_argument(
        "--debug-memory",
        action="store_true",
        help="Print CUDA memory and model input tensor shapes around each forward pass",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary success in plot (default: 0.5)",
    )
    args = parser.parse_args()

    task = read_task(args.task, args.task_file)
    videos = resolve_videos(args.input_dir, args.video, args.pattern)
    out_dir = resolve_output_dir(args.input_dir, videos, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = LocalRewardRunner(model_path=args.model_path, debug_memory=bool(args.debug_memory))

    summaries = []
    for idx, video_path in enumerate(videos):
        frames = load_frames_input(
            str(video_path),
            fps=float(args.fps),
            max_frames=int(args.max_frames),
            resize_max_side=int(args.resize_max_side),
        )
        print(f"Loaded frames for {video_path.name}: shape={frames.shape}, dtype={frames.dtype}")
        rewards, success_probs = runner.compute_rewards_per_frame(
            video_frames=frames,
            task=task,
            sample_id=str(idx),
        )

        out_path = out_dir / f"{video_path.stem}_rewards.npy"
        np.save(str(out_path), rewards)
        success_path = out_dir / f"{video_path.stem}_rewards_success_probs.npy"
        np.save(str(success_path), success_probs)

        show_success = success_probs.size > 0 and success_probs.size == rewards.size
        success_binary = (success_probs > float(args.success_threshold)).astype(np.int32) if show_success else None
        fig = create_combined_progress_success_plot(
            progress_pred=rewards,
            num_frames=int(frames.shape[0]),
            success_binary=success_binary,
            success_probs=success_probs if show_success else None,
            success_labels=None,
            title=f"Progress/Success — {video_path.name}",
        )
        plot_path = out_dir / f"{video_path.stem}_rewards_progress_success.png"
        fig.savefig(str(plot_path), dpi=200)
        plt.close(fig)

        summaries.append(
            {
                "video": str(video_path),
                "task": task,
                "num_frames": int(frames.shape[0]),
                "frames_shape": list(frames.shape),
                "out_rewards": str(out_path),
                "out_success_probs": str(success_path),
                "out_plot": str(plot_path),
                "reward_min": float(np.min(rewards)) if rewards.size else None,
                "reward_max": float(np.max(rewards)) if rewards.size else None,
                "reward_mean": float(np.mean(rewards)) if rewards.size else None,
            }
        )

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"processed_videos": len(summaries), "summary_file": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
