"""Data-parallel SGD worker and launcher.

Each worker:
  1. joins the TCP ring and hits the barrier,
  2. builds the model from a shared seed (identical init on all workers),
  3. trains on its strided data shard: forward/backward on a local batch,
     ring all-reduce of gradients (sum), divide by world size, SGD step,
  4. writes its loss trajectory to the results directory.

The launcher spawns N subprocesses and waits for them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn

from .collectives import allreduce_sum, broadcast
from .data import make_regression_data, shard_indices
from .transport import barrier, make_ring

__all__ = ["worker_main", "launch"]


def build_model(in_features: int, seed: int) -> nn.Module:
    gen = torch.Generator().manual_seed(seed)
    model = nn.Sequential(nn.Linear(in_features, 64), nn.ReLU(), nn.Linear(64, 1))
    for p in model.parameters():
        p.data = torch.randn(p.shape, generator=gen) * 0.1
    return model


def worker_main(args: argparse.Namespace) -> None:
    # One thread per worker: N workers already oversubscribe a small VM;
    # letting each spawn torch's default thread pool would thrash.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rank, world_size = args.rank, args.world_size

    inbound, outbound = make_ring(rank, world_size, args.base_port)
    try:
        barrier(inbound, outbound, rank)

        X, y, _ = make_regression_data(
            num_samples=args.num_samples,
            in_features=args.in_features,
            noise_std=args.noise_std,
            seed=args.data_seed,
        )
        shard = shard_indices(args.num_samples, rank, world_size)
        Xs, ys = X[shard], y[shard]

        model = build_model(args.in_features, args.seed)
        # Broadcast rank 0's params so init is identical even if a future
        # code path diverges in RNG usage.
        for p in model.parameters():
            p.data = broadcast(p.data, rank, world_size, inbound, outbound, src=0)

        opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()
        local_batch = args.batch_size // world_size
        assert args.batch_size % world_size == 0, "batch size must divide by world size"
        steps = len(shard) // local_batch

        losses: list[float] = []
        step_times: list[float] = []
        for step in range(steps):
            batch = slice(step * local_batch, (step + 1) * local_batch)
            t0 = time.perf_counter()
            opt.zero_grad()
            # Sum (not mean): the all-reduce sums across workers, then we
            # divide once by the global batch size -> exact global mean.
            loss = loss_fn(model(Xs[batch]), ys[batch]) * local_batch
            loss.backward()
            # Gradient bucketing: one all-reduce per step instead of one per
            # parameter. Production DDP does the same; per-message TCP/Python
            # overhead dominates for small models.
            grads = [p.grad.data for p in model.parameters() if p.grad is not None]
            shapes = [g.shape for g in grads]
            bucket = torch.cat([g.flatten() for g in grads])
            bucket = allreduce_sum(bucket, rank, world_size, inbound, outbound)
            bucket = bucket / args.batch_size
            offset = 0
            idx = 0
            for p in model.parameters():
                if p.grad is not None:
                    n = p.grad.numel()
                    p.grad.data = bucket[offset:offset + n].reshape(shapes[idx])
                    offset += n
                    idx += 1
            opt.step()
            step_times.append(time.perf_counter() - t0)
            # Global loss for the trajectory: sum over the shard, all-reduce,
            # divide once -> identical quantity in 1-worker and N-worker runs.
            with torch.no_grad():
                local_sum = loss_fn(model(Xs), ys) * len(shard)
                global_loss = allreduce_sum(
                    local_sum, rank, world_size, inbound, outbound
                ) / args.num_samples
                losses.append(float(global_loss.item()))

        os.makedirs(args.results_dir, exist_ok=True)
        with open(os.path.join(args.results_dir, f"rank_{rank}.json"), "w") as f:
            json.dump(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "losses": losses,
                    "mean_step_ms": 1000 * sum(step_times) / len(step_times),
                    "num_steps": steps,
                },
                f,
            )
    finally:
        inbound.close()
        outbound.close()


def launch(args: argparse.Namespace) -> int:
    """Spawn world_size worker subprocesses and wait for all of them."""
    procs = []
    for rank in range(args.world_size):
        cmd = [
            sys.executable, "-m", "ddp.train", "worker",
            "--rank", str(rank),
            "--world-size", str(args.world_size),
            "--base-port", str(args.base_port),
            "--seed", str(args.seed),
            "--data-seed", str(args.data_seed),
            "--num-samples", str(args.num_samples),
            "--in-features", str(args.in_features),
            "--noise-std", str(args.noise_std),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--results-dir", args.results_dir,
        ]
        procs.append(subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    failures = 0
    for rank, proc in enumerate(procs):
        code = proc.wait()
        if code != 0:
            print(f"worker {rank} exited with code {code}", file=sys.stderr)
            failures += 1
    return failures


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Data-parallel SGD over a TCP ring")
    sub = p.add_subparsers(dest="command", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--rank", type=int, required=True)
    w.add_argument("--world-size", type=int, required=True)
    w.add_argument("--base-port", type=int, default=29400)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--data-seed", type=int, default=0)
    w.add_argument("--num-samples", type=int, default=4096)
    w.add_argument("--in-features", type=int, default=20)
    w.add_argument("--noise-std", type=float, default=0.1)
    w.add_argument("--batch-size", type=int, default=256)
    w.add_argument("--lr", type=float, default=0.05)
    w.add_argument("--results-dir", default="results")
    l = sub.add_parser("launch")
    l.add_argument("--world-size", type=int, required=True)
    l.add_argument("--base-port", type=int, default=29400)
    l.add_argument("--seed", type=int, default=0)
    l.add_argument("--data-seed", type=int, default=0)
    l.add_argument("--num-samples", type=int, default=4096)
    l.add_argument("--in-features", type=int, default=20)
    l.add_argument("--noise-std", type=float, default=0.1)
    l.add_argument("--batch-size", type=int, default=256)
    l.add_argument("--lr", type=float, default=0.05)
    l.add_argument("--results-dir", default="results")
    return p


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    if parsed.command == "worker":
        worker_main(parsed)
    else:
        sys.exit(launch(parsed))
