"""The core data-parallelism proof: N workers with batch B/N each must follow
the same loss trajectory as 1 worker with batch B, given identical seeds.
Strided sharding makes the union of workers' batches exactly the
single-worker batch, and gradients are summed then divided once."""
import json
import os
import subprocess
import sys

import pytest

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_training(world_size, base_port, results_dir, batch_size=256, num_samples=4096):
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "ddp.train", "launch",
           "--world-size", str(world_size),
           "--base-port", str(base_port),
           "--batch-size", str(batch_size),
           "--num-samples", str(num_samples),
           "--results-dir", results_dir,
           "--seed", "0", "--data-seed", "0"]
    proc = subprocess.run(cmd, cwd=PROJ_DIR, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"launch failed:\n{proc.stdout}\n{proc.stderr}"
    losses = []
    for rank in range(world_size):
        with open(os.path.join(results_dir, f"rank_{rank}.json")) as f:
            losses.append(json.load(f)["losses"])
    return losses


def test_distributed_matches_single_worker(tmp_path):
    # Small enough to run fast, large enough to be meaningful.
    dir1 = str(tmp_path / "w1")
    dir4 = str(tmp_path / "w4")
    losses_1 = run_training(1, 31001, dir1, batch_size=128, num_samples=2048)
    losses_4 = run_training(4, 32001, dir4, batch_size=128, num_samples=2048)
    assert len(losses_1) == 1 and len(losses_4) == 4
    traj_1 = losses_1[0]
    traj_4 = losses_4[0]
    assert len(traj_1) == len(traj_4)
    # Same optimization trajectory up to fp summation order.
    for i, (a, b) in enumerate(zip(traj_1, traj_4)):
        assert abs(a - b) < 1e-3 * max(1.0, abs(a)), \
            f"step {i}: single={a:.6f} distributed={b:.6f}"
    # And it actually learned something.
    assert traj_1[-1] < traj_1[0] * 0.5
    # Workers agree with each other to fp precision.
    for r in range(1, 4):
        for i, (a, b) in enumerate(zip(traj_4, losses_4[r])):
            assert abs(a - b) < 1e-6, f"worker {r} diverged at step {i}"
