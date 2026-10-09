# Distributed Training Engine

<p align="justify">
From-scratch data-parallel SGD over TCP in Python: N worker processes train identical model replicas on sharded data, synchronize gradients with a bandwidth-optimal ring all-reduce, and apply the averaged update. No PyTorch DDP, no NCCL, no framework collectives — the transport, the all-reduce, the gradient bucketing, and the training loop are all implemented here and verified against single-worker training.
</p>

## How it works

<p align="justify">
Each worker listens on <code>base_port + rank</code> and connects to the next rank, forming a directed TCP ring. After a token-ring barrier, every worker builds an identical model (shared seed plus a broadcast of rank 0's parameters) and trains on its strided data shard: worker <code>r</code> owns samples <code>r, r+N, r+2N, ...</code>. The striding is deliberate — the union of all workers' batches at step <code>s</code> is exactly the single-worker batch, which is what makes the equivalence test meaningful.
</p>

<p align="justify">
Each step, a worker runs forward/backward on its local batch, flattens all gradients into one bucket, ring all-reduces the bucket (sum), divides once by the global batch size, and takes an SGD step. Bucketing matters: one all-reduce per step instead of one per parameter, which is the same optimization PyTorch DDP uses, because per-message TCP and Python overhead dominates for small models.
</p>

### Ring all-reduce

<p align="justify">
The tensor is split into N chunks. In the reduce-scatter phase (N-1 steps), each worker sends one chunk right, receives one from the left, and accumulates — leaving worker <code>r</code> holding fully-reduced chunk <code>(r+1)%N</code>. In the allgather phase (N-1 steps), reduced chunks circulate until every worker holds the full sum. Per-worker data moved is <code>2(N-1)/N</code> times the tensor size, which is bandwidth-optimal: independent of N for large N. Reduction runs in fp32 regardless of input dtype.
</p>

### The Nagle incident

<p align="justify">
The first all-reduce benchmark showed 42ms for a 22KB tensor across 4 workers on localhost — absurd for loopback TCP. The cause was Nagle's algorithm: small collectives messages were being held for delayed ACKs, adding milliseconds per send/recv round-trip. Setting <code>TCP_NODELAY</code> on every socket dropped it to 0.65ms, a 65x improvement from one socket option. This is the kind of bug that only shows up when you measure, which is why the benchmark scripts exist.
</p>

## Verification

<p align="justify">
Correctness is established by three layers of tests, all running real worker processes over real TCP sockets (no mocks):
</p>

- **Collectives** (`tests/test_collectives.py`): ring all-reduce against analytically known sums for 2, 3, and 4 workers (1000 elements, deliberately not divisible by 3, exercising chunk padding); broadcast from rank 0 for 2–4 workers; tensor wire-format roundtrip.
- **Equivalence** (`tests/test_equivalence.py`): the central proof. One worker with global batch B and four workers with local batch B/4, identical seeds, produce the same loss trajectory step-by-step (tolerance accounts only for fp summation order), all four workers agree with each other to 1e-6, and the loss actually halves — the model learns.
- **Transport**: framed message integrity, connection timeouts, barrier correctness.

## Benchmark

<p align="justify">
Measured with <code>scripts/demo.py</code> on a 2-vCPU cloud VM (weak scaling: fixed 64-sample batch and 2048 samples per worker, tiny 5.5K-parameter MLP, 32 steps):
</p>

| workers | global batch | ms/step | efficiency vs 1 worker |
|---|---|---|---|
| 1 | 64 | 0.5 | 1.00x |
| 2 | 128 | 0.9 | 0.55x |
| 4 | 256 | 6.1 | 0.08x |

<p align="justify">
The 1→2 worker number is the honest signal: per-step time roughly doubles because communication (two all-reduces per step: gradients plus the global loss) is significant next to a 5.5K-parameter model's compute. The 4-worker number is not a scaling claim — four processes on two vCPUs is oversubscribed hardware, and it is reported as measured rather than hidden. On hardware with one core per worker and a larger model, compute would dominate and the efficiency would be substantially higher; the all-reduce microbenchmark (0.65ms for 22KB across 4 workers) bounds the communication floor.
</p>

## Limitations

<p align="justify">
This is a fundamentals implementation, stated plainly. It is CPU-only with no CUDA/NCCL path — the all-reduce algorithm is identical either way, but GPU interconnects are a different performance regime. There is no fault tolerance: a dead worker hangs the ring, and there is no checkpointing or elastic re-sharding. Only synchronous SGD is implemented (no async, no gradient compression, no mixed precision). The transport assumes a flat network where every worker can open TCP to every other worker's successor — no NAT traversal, no multi-node discovery beyond explicit ports.
</p>

## Project structure

```text
distributed-training-engine/
├── ddp/
│   ├── transport.py    # TCP ring formation, framed tensor messages, barrier
│   ├── collectives.py  # ring all-reduce (sum), broadcast
│   ├── data.py         # synthetic regression data, strided sharding
│   └── train.py        # worker train loop + process launcher
├── tests/
│   ├── test_collectives.py  # all-reduce/broadcast over real TCP, 2-4 workers
│   ├── test_equivalence.py  # N-worker vs 1-worker trajectory match
│   └── ring_worker.py       # test worker entrypoint
├── scripts/
│   └── demo.py         # end-to-end weak-scaling demo
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt

# 4 workers, each on base_port + rank starting at 29400
python -m ddp.train launch --world-size 4 --base-port 29400

# Weak-scaling demo
python scripts/demo.py --workers 1 2 4

# Tests (spawns real worker processes; takes ~1 min)
pytest -q
```

## Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--world-size` | (required) | Number of worker processes |
| `--base-port` | 29400 | Worker `rank` listens on `base_port + rank` |
| `--batch-size` | 256 | Global batch (must divide by world size) |
| `--num-samples` | 4096 | Synthetic dataset size |
| `--lr` | 0.05 | SGD learning rate |
| `--seed` / `--data-seed` | 0 | Model init / dataset seeds |

## License

MIT — see [LICENSE](LICENSE).
