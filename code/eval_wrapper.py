"""Thin subprocess wrapper around Xiaomi's real client-server
eval_robocasa365 pipeline (vendor/Xiaomi-Robotics-1/scripts/deploy.sh +
scripts/launch_robocasa365.sh) -- their eval is not an importable Python
API, so this wraps the real shell invocation and parses the real
summary.json it produces, rather than re-implementing their eval logic."""
import json
import subprocess
from pathlib import Path

XIAOMI_REPO = Path("/workspace/xr1-m7-submission/vendor/Xiaomi-Robotics-1")


def run_eval(checkpoint_path: str, output_dir: str, num_gpus: int = 1,
             extra_args: list[str] | None = None) -> dict:
    """Starts a single-GPU server, runs the client against it, and returns
    the parsed summary.json. Raises CalledProcessError if either the deploy
    or launch script exits non-zero -- callers should not silently treat a
    failed eval run as a 0% result."""
    subprocess.run(
        ["bash", "scripts/deploy.sh", checkpoint_path, str(num_gpus), str(num_gpus)],
        cwd=XIAOMI_REPO, check=True,
    )
    cmd = ["bash", "scripts/launch_robocasa365.sh", str(num_gpus), output_dir, checkpoint_path]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=XIAOMI_REPO, check=True, env={"CONDA_ENV": "robocasa_365"})

    summary_files = list(Path(output_dir).glob("*/summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary.json produced under {output_dir}")
    with open(summary_files[0]) as f:
        return json.load(f)
