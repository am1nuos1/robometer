#!/usr/bin/env python3
"""Run batch inference over a configurable parameter sweep."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


SWEEP_KEYS = ("quantization", "max_frames", "resize_max_side", "fps")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | None, root: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return root / p


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def expand_values(spec: Any, default_count: int = 1) -> list[Any]:
    """Expand a literal list or a start/step/count range."""
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        if "values" in spec:
            if not isinstance(spec["values"], list):
                raise ValueError("'values' must be a list")
            return spec["values"]
        start = spec["start"]
        step = spec["step"]
        count = int(spec.get("count", default_count))
        if count < 1:
            raise ValueError("range count must be >= 1")
        if count == 1:
            return [start]
        if step == 0:
            raise ValueError("range step cannot be 0 when count is greater than 1")
        return [start + step * i for i in range(count)]
    return [spec]


def expand_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = config["sweep"]
    default_count = int(config.get("range_count", 1))
    value_lists = {key: expand_values(sweep.get(key), default_count) for key in SWEEP_KEYS}
    missing = [key for key, values in value_lists.items() if values == [None]]
    if missing:
        raise ValueError(f"Missing sweep keys: {', '.join(missing)}")

    combos = []
    for values in itertools.product(*(value_lists[key] for key in SWEEP_KEYS)):
        combos.append(dict(zip(SWEEP_KEYS, values)))
    return combos


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def safe_value(value: Any) -> str:
    text = bool_text(value)
    return text.replace(".", "p").replace("/", "_").replace(" ", "")


def run_name(combo: dict[str, Any]) -> str:
    return (
        f"q{safe_value(combo['quantization'])}"
        f"_mf{safe_value(combo['max_frames'])}"
        f"_resize{safe_value(combo['resize_max_side'])}"
        f"_fps{safe_value(combo['fps'])}"
    )


def set_quantization(config_path: Path, enabled: bool) -> str:
    """Patch the model config.yaml quantization flag and return original text."""
    original = config_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    in_model = False
    changed = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if line.startswith("model:"):
            in_model = True
            continue
        if in_model and line and not line.startswith((" ", "\t")):
            in_model = False
        if in_model and stripped.startswith("quantization:"):
            indent = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            lines[idx] = f"{indent}quantization: {bool_text(enabled).capitalize()}{newline}"
            changed = True
            break

    if not changed:
        raise ValueError(f"Could not find model.quantization in {config_path}")

    config_path.write_text("".join(lines), encoding="utf-8")
    return original


def restore_file(config_path: Path, original_text: str) -> None:
    config_path.write_text(original_text, encoding="utf-8")


def build_command(config: dict[str, Any], combo: dict[str, Any], out_dir: Path, root: Path) -> list[str]:
    python_executable = resolve_path(config["python_executable"], root)
    script_path = resolve_path(config["script_path"], root)
    input_dir = resolve_path(config["input_dir"], root)
    task_file = resolve_path(config["task_file"], root)

    cmd = [
        str(python_executable),
        str(script_path),
        "--model-path",
        str(config["model_path"]),
        "--input-dir",
        str(input_dir),
        "--pattern",
        str(config.get("pattern", "*.mp4")),
        "--task-file",
        str(task_file),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(combo["fps"]),
        "--max-frames",
        str(combo["max_frames"]),
        "--resize-max-side",
        str(combo["resize_max_side"]),
        "--success-threshold",
        str(config.get("success_threshold", 0.5)),
    ]
    if config.get("debug_memory", False):
        cmd.append("--debug-memory")
    return cmd


def write_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    base_fieldnames = [
        "index",
        "status",
        "returncode",
        "duration_sec",
        "quantization",
        "max_frames",
        "resize_max_side",
        "fps",
        "out_dir",
        "log_file",
        "error_type",
        "error_excerpt",
    ]
    extra_fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in base_fieldnames and key not in {"command", "total"}
        }
    )
    fieldnames = base_fieldnames + extra_fieldnames
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown_table(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    if not rows:
        return

    video_stems = sorted(config.get("expected_success", {}).keys())
    fieldnames = [
        "index",
        "status",
        "quantization",
        "max_frames",
        "resize_max_side",
        "fps",
        "pairwise_label_accuracy",
        "success_failure_margin",
    ]
    for stem in video_stems:
        fieldnames.append(f"{stem}_reward_mean")
    fieldnames.extend(["error_type", "error_excerpt"])

    lines = []
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for row in rows:
        values = [markdown_cell(row.get(key, "")) for key in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    text = text.replace("|", "\\|")
    if len(text) > 160:
        text = text[:157] + "..."
    return text


def run_one(
    index: int,
    total: int,
    config: dict[str, Any],
    combo: dict[str, Any],
    output_root: Path,
    root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    name = run_name(combo)
    out_dir = output_root / name
    log_file = output_root / f"{name}.log"
    cmd = build_command(config, combo, out_dir, root)

    result = {
        "index": index,
        "total": total,
        "status": "dry_run" if dry_run else "pending",
        "returncode": None,
        "duration_sec": 0.0,
        "out_dir": str(out_dir),
        "log_file": str(log_file),
        "command": cmd,
        **combo,
    }

    print(f"[{index}/{total}] {name}")
    print(" ".join(cmd))
    if dry_run:
        return result

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    env = os.environ.copy()
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=config.get("timeout_seconds"),
            text=True,
        )
    result["returncode"] = proc.returncode
    result["duration_sec"] = round(time.time() - start, 3)
    if proc.returncode == 0:
        result["status"] = "ok"
        result["error_type"] = ""
        result["error_excerpt"] = ""
        result.update(read_run_metrics(out_dir, config))
    else:
        error_type, error_excerpt = classify_failure(log_file)
        result["status"] = error_type
        result["error_type"] = error_type
        result["error_excerpt"] = error_excerpt
    return result


def read_run_metrics(out_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read per-video metrics from batch_summary.json and .npy outputs."""
    summary_path = out_dir / "batch_summary.json"
    if not summary_path.exists():
        return {"metrics_error": f"missing {summary_path.name}"}

    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"metrics_error": str(exc)}

    expected_success = config.get("expected_success", {})
    metrics: dict[str, Any] = {}
    success_rewards = []
    failure_rewards = []
    correct = 0
    labeled = 0

    for item in summary:
        stem = Path(item["video"]).stem
        reward_mean = item.get("reward_mean")
        metrics[f"{stem}_frames"] = item.get("num_frames")
        metrics[f"{stem}_reward_mean"] = reward_mean
        metrics[f"{stem}_reward_last"] = read_last_npy_value(item.get("out_rewards"))
        metrics[f"{stem}_success_prob_last"] = read_last_npy_value(item.get("out_success_probs"))

        if stem in expected_success and reward_mean is not None:
            is_success = bool(expected_success[stem])
            metrics[f"{stem}_label"] = "success" if is_success else "failure"
            if is_success:
                success_rewards.append(float(reward_mean))
            else:
                failure_rewards.append(float(reward_mean))

    for success_reward in success_rewards:
        for failure_reward in failure_rewards:
            labeled += 1
            if success_reward > failure_reward:
                correct += 1

    if success_rewards:
        metrics["expected_success_reward_mean"] = round(sum(success_rewards) / len(success_rewards), 6)
    if failure_rewards:
        metrics["expected_failure_reward_mean"] = round(sum(failure_rewards) / len(failure_rewards), 6)
    if success_rewards and failure_rewards:
        metrics["success_failure_margin"] = round(
            metrics["expected_success_reward_mean"] - metrics["expected_failure_reward_mean"], 6
        )
    if labeled:
        metrics["pairwise_label_accuracy"] = round(correct / labeled, 6)
    return metrics


def read_last_npy_value(path: str | None) -> float | None:
    if not path:
        return None
    try:
        import numpy as np

        arr = np.load(path)
    except Exception:
        return None
    if arr.size == 0:
        return None
    return round(float(arr.reshape(-1)[-1]), 6)


def classify_failure(log_file: Path) -> tuple[str, str]:
    """Classify common inference failures from the subprocess log."""
    try:
        text = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return "failed", f"Could not read log: {exc}"

    lowered = text.lower()
    if "torch.outofmemoryerror" in lowered or "cuda out of memory" in lowered:
        return "oom", extract_error_excerpt(text, ["OutOfMemoryError", "CUDA out of memory"])
    if "traceback (most recent call last)" in lowered:
        return "failed", extract_error_excerpt(text, ["Traceback (most recent call last)"])
    return "failed", text[-1000:].replace("\n", " ")


def extract_error_excerpt(text: str, markers: Iterable[str]) -> str:
    """Return a compact one-line excerpt around the first matching marker."""
    positions = [text.find(marker) for marker in markers if text.find(marker) >= 0]
    if not positions:
        return text[-1000:].replace("\n", " ")
    start = max(0, min(positions) - 300)
    end = min(len(text), min(positions) + 700)
    return text[start:end].replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="debug_tools/inference_sweep/sweep_config.json",
        help="Path to sweep config JSON",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument("--clean", action="store_true", help="Delete output_root before running")
    args = parser.parse_args()

    root = repo_root()
    config_path = resolve_path(args.config, root)
    config = load_json(config_path)
    output_root = resolve_path(config["output_root"], root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    combos = expand_sweep(config)
    summary_jsonl = output_root / "summary.jsonl"
    summary_csv = output_root / "summary.csv"
    summary_table = output_root / "summary_table.md"
    if summary_jsonl.exists() and not args.dry_run:
        summary_jsonl.unlink()

    quant_config_path = resolve_path(config.get("quantization_config_path"), root)
    rows: list[dict[str, Any]] = []
    original_quant_config = None

    try:
        if quant_config_path and not args.dry_run:
            original_quant_config = quant_config_path.read_text(encoding="utf-8")

        for index, combo in enumerate(combos, start=1):
            patched_text = None
            try:
                if quant_config_path and not args.dry_run:
                    patched_text = set_quantization(quant_config_path, bool(combo["quantization"]))
                row = run_one(index, len(combos), config, combo, output_root, root, args.dry_run)
            except subprocess.TimeoutExpired as exc:
                row = {
                    "index": index,
                    "total": len(combos),
                    "status": "timeout",
                    "returncode": None,
                    "duration_sec": config.get("timeout_seconds"),
                    "out_dir": str(output_root / run_name(combo)),
                    "log_file": str(output_root / f"{run_name(combo)}.log"),
                    "error_type": "timeout",
                    "error_excerpt": f"Timed out after {config.get('timeout_seconds')} seconds",
                    "command": getattr(exc, "cmd", []),
                    **combo,
                }
            finally:
                if quant_config_path and patched_text is not None:
                    restore_file(quant_config_path, patched_text)

            rows.append(row)
            if not args.dry_run:
                write_jsonl(summary_jsonl, row)
                write_csv(summary_csv, rows)
                write_markdown_table(summary_table, rows, config)
            if row["status"] != "ok" and not config.get("continue_on_error", True):
                break

    finally:
        if quant_config_path and original_quant_config is not None:
            restore_file(quant_config_path, original_quant_config)

    if not args.dry_run:
        print(f"Wrote {summary_csv}")
        print(f"Wrote {summary_jsonl}")
        print(f"Wrote {summary_table}")
    return 0 if all(row["status"] in ("ok", "dry_run") for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
