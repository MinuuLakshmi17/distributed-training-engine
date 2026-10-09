"""Worker entrypoint for collective tests. Usage:
python ring_worker.py --mode allreduce|broadcast --rank R --world-size N ...
Writes results/<mode>_rank_<R>.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ddp.collectives import allreduce_sum, broadcast
from ddp.transport import barrier, make_ring


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--base-port", type=int, required=True)
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()

    torch.manual_seed(0)
    inbound, outbound = make_ring(args.rank, args.world_size, args.base_port)
    try:
        barrier(inbound, outbound, args.rank)
        if args.mode == "allreduce":
            # Each rank contributes (rank+1) * ones; uneven size tests padding.
            local = torch.full((1000,), float(args.rank + 1))
            result = allreduce_sum(local, args.rank, args.world_size, inbound, outbound)
            expected = sum(range(1, args.world_size + 1))
            payload = {"values": result.tolist(),
                       "expected": float(expected)}
        elif args.mode == "broadcast":
            src_tensor = torch.arange(50, dtype=torch.float32) * 2.5 if args.rank == 0 else None
            result = broadcast(src_tensor, args.rank, args.world_size, inbound, outbound, src=0)
            payload = {"values": result.tolist(),
                       "expected": (torch.arange(50, dtype=torch.float32) * 2.5).tolist()}
        else:
            raise ValueError(args.mode)
        with open(os.path.join(args.results_dir, f"{args.mode}_rank_{args.rank}.json"), "w") as f:
            json.dump(payload, f)
    finally:
        inbound.close()
        outbound.close()


if __name__ == "__main__":
    main()
