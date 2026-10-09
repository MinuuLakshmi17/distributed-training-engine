"""End-to-end demo: train a tiny MLP with data-parallel SGD and report
weak scaling (fixed work per worker, varying worker count).

Usage: python scripts/demo.py [--workers 1 2 4] [--base-port 29600]
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_once(world_size, base_port, results_dir, args):
    cmd = [sys.executable, "-m", "ddp.train", "launch",
           "--world-size", str(world_size),
           "--base-port", str(base_port),
           "--batch-size", str(args.batch_size * world_size),
           "--num-samples", str(args.num_samples * world_size),
           "--results-dir", results_dir,
           "--seed", "0", "--data-seed", "0",
           "--lr", str(args.lr)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=PROJ_DIR, capture_output=True, text=True, timeout=900)
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"world_size={world_size} failed:\n{proc.stdout}\n{proc.stderr}")
    with open(os.path.join(results_dir, "rank_0.json")) as f:
        result = json.load(f)
    return result, wall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--base-port", type=int, default=29600)
    p.add_argument("--batch-size", type=int, default=64,
                   help="per-worker batch size (global batch scales with workers)")
    p.add_argument("--num-samples", type=int, default=2048,
                   help="per-worker samples (weak scaling: fixed work per worker)")
    p.add_argument("--lr", type=float, default=0.05)
    args = p.parse_args()

    print(f"{'workers':>8} {'global_batch':>13} {'steps':>6} "
          f"{'final_loss':>11} {'ms/step':>9} {'wall(s)':>8}")
    baseline_ms = None
    for i, w in enumerate(args.workers):
        results_dir = os.path.join(PROJ_DIR, "results", f"demo_w{w}")
        result, wall = run_once(w, args.base_port + i * 100, results_dir, args)
        ms_step = result["mean_step_ms"]
        baseline_ms = baseline_ms or ms_step
        print(f"{w:>8} {args.batch_size * w:>13} {result['num_steps']:>6} "
              f"{result['losses'][-1]:>11.4f} {ms_step:>9.1f} {wall:>8.1f} "
              f"(efficiency vs 1 worker: {baseline_ms / ms_step:>5.2f}x)")


if __name__ == "__main__":
    main()
