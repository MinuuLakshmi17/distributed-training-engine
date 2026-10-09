"""Collective communication over the TCP ring.

Ring all-reduce (bandwidth-optimal): with p workers each holding a tensor,
the tensor is split into p chunks; p-1 reduce-scatter steps leave worker r
holding the fully-reduced chunk r, then p-1 allgather steps replicate every
chunk everywhere. Total data moved per worker: 2*(p-1)/p * tensor_size.
"""

from __future__ import annotations

import torch

from .transport import Peer

__all__ = ["allreduce_sum", "broadcast"]


def _split_chunks(tensor: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    flat = tensor.detach().contiguous().flatten()
    chunk_size = (flat.numel() + world_size - 1) // world_size
    padded_len = chunk_size * world_size
    if flat.numel() < padded_len:
        flat = torch.cat([flat, torch.zeros(padded_len - flat.numel(), dtype=flat.dtype)])
    return list(flat.chunk(world_size))


def allreduce_sum(
    tensor: torch.Tensor,
    rank: int,
    world_size: int,
    inbound: Peer,
    outbound: Peer,
) -> torch.Tensor:
    """Sum ``tensor`` across all workers; every worker gets the full sum."""
    if world_size == 1:
        return tensor.detach().clone()
    orig_shape, orig_numel = tensor.shape, tensor.numel()
    dtype = tensor.dtype
    chunks = _split_chunks(tensor.float(), world_size)

    # Reduce-scatter: after step s, chunk[(r - s - 1) % p] accumulates.
    for step in range(world_size - 1):
        send_idx = (rank - step) % world_size
        recv_idx = (rank - step - 1) % world_size
        outbound.send_tensor(chunks[send_idx])
        received = inbound.recv_tensor()
        chunks[recv_idx] = chunks[recv_idx] + received.to(chunks[recv_idx].dtype)

    # Allgather: reduce-scatter leaves rank r owning reduced chunk (r+1)%p,
    # so step s sends (r+1-s)%p (owned, then previously received) and
    # receives the next unseen chunk (r-s)%p.
    for step in range(world_size - 1):
        send_idx = (rank + 1 - step) % world_size
        recv_idx = (rank - step) % world_size
        outbound.send_tensor(chunks[send_idx])
        chunks[recv_idx] = inbound.recv_tensor().to(chunks[recv_idx].dtype)

    result = torch.cat(chunks)[:orig_numel].reshape(orig_shape)
    return result.to(dtype)


def broadcast(
    tensor: torch.Tensor | None,
    rank: int,
    world_size: int,
    inbound: Peer,
    outbound: Peer,
    src: int = 0,
) -> torch.Tensor:
    """Send ``tensor`` from ``src`` to every worker along the ring."""
    if world_size == 1:
        assert tensor is not None
        return tensor.detach().clone()
    # Distance ahead of src in ring order.
    dist = (rank - src) % world_size
    if dist == 0:
        assert tensor is not None
        current = tensor.detach().clone()
    else:
        current = inbound.recv_tensor()
    if dist != world_size - 1:
        outbound.send_tensor(current)
    return current
