"""Spawns real worker processes to verify collectives over TCP."""
import json
import os
import random
import subprocess
import sys

import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def run_ring(world_size, mode, extra_args=(), timeout=120):
    """Spawn world_size workers; each writes results/mode_rank_<r>.json."""
    base_port = random.randint(30000, 50000)
    results_dir = os.path.join(TEST_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    # Clean stale results for this mode.
    for f in os.listdir(results_dir):
        if f.startswith(mode + "_rank_"):
            os.remove(os.path.join(results_dir, f))
    procs = []
    for rank in range(world_size):
        cmd = [sys.executable, os.path.join(TEST_DIR, "ring_worker.py"),
               "--mode", mode, "--rank", str(rank),
               "--world-size", str(world_size), "--base-port", str(base_port),
               "--results-dir", results_dir, *extra_args]
        procs.append(subprocess.Popen(cmd))
    for rank, proc in enumerate(procs):
        code = proc.wait(timeout=timeout)
        assert code == 0, f"worker {rank} (mode={mode}) exited with code {code}"
    out = []
    for rank in range(world_size):
        with open(os.path.join(results_dir, f"{mode}_rank_{rank}.json")) as f:
            out.append(json.load(f))
    return out
