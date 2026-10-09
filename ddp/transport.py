"""TCP transport for the training ring.

Each worker listens on ``base_port + rank`` and connects to
``base_port + (rank + 1) % world_size``, forming a directed ring.
Messages are length-prefixed frames; tensors are sent as
``[dtype][ndim][shape...][raw bytes]``.
"""

from __future__ import annotations

import socket
import struct
import time

import torch

__all__ = ["Peer", "make_ring", "barrier"]

_HEADER_FMT = ">Q"  # uint64 payload length, big-endian
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class Peer:
    """One TCP connection with framed send/recv."""

    def __init__(self, sock: socket.socket):
        # Disable Nagle: collectives are latency-sensitive small messages,
        # and Nagle's delayed ACKs interact pathologically with send/recv
        # ping-pong patterns (40ms+ per all-reduce step without this).
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock

    def send_bytes(self, payload: bytes) -> None:
        self.sock.sendall(struct.pack(_HEADER_FMT, len(payload)) + payload)

    def _recv_exactly(self, n: int) -> bytes:
        chunks = []
        while n > 0:
            chunk = self.sock.recv(n)
            if not chunk:
                raise ConnectionError("peer closed the connection mid-frame")
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def recv_bytes(self) -> bytes:
        (length,) = struct.unpack(_HEADER_FMT, self._recv_exactly(_HEADER_SIZE))
        if length > 512 * 1024 * 1024:
            raise ValueError(f"frame too large: {length} bytes")
        return self._recv_exactly(length)

    def send_tensor(self, tensor: torch.Tensor) -> None:
        t = tensor.detach().contiguous().cpu()
        dtype_name = str(t.dtype).replace("torch.", "")
        header = (
            bytes([len(dtype_name)])
            + dtype_name.encode()
            + bytes([t.ndim])
            + struct.pack(f">{t.ndim}Q", *t.shape)
        )
        self.send_bytes(header + t.numpy().tobytes())

    def recv_tensor(self) -> torch.Tensor:
        payload = self.recv_bytes()
        dtype_len = payload[0]
        dtype_name = payload[1 : 1 + dtype_len].decode()
        ndim = payload[1 + dtype_len]
        off = 2 + dtype_len
        shape = struct.unpack(f">{ndim}Q", payload[off : off + 8 * ndim])
        raw = payload[off + 8 * ndim :]
        dtype = getattr(torch, dtype_name)
        # frombuffer shares memory; clone to own it.
        arr = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
        return arr.reshape(shape)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


def make_ring(
    rank: int,
    world_size: int,
    base_port: int,
    host: str = "127.0.0.1",
    timeout_s: float = 60.0,
) -> tuple[Peer, Peer]:
    """Form the ring. Returns ``(inbound, outbound)`` peers.

    Listens on ``base_port + rank``; connects to the next rank's listener
    with retries. Raises ``TimeoutError`` if the ring doesn't form in time.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, base_port + rank))
    listener.listen(1)

    next_rank = (rank + 1) % world_size
    outbound_sock: socket.socket | None = None
    deadline = time.time() + timeout_s
    while outbound_sock is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((host, base_port + next_rank))
            s.settimeout(None)
            outbound_sock = s
        except OSError:
            if time.time() > deadline:
                listener.close()
                raise TimeoutError(f"rank {rank}: could not connect to rank {next_rank}")
            time.sleep(0.05)

    listener.settimeout(max(0.1, deadline - time.time()))
    try:
        inbound_sock, _ = listener.accept()
    except socket.timeout as exc:
        outbound_sock.close()
        listener.close()
        raise TimeoutError(f"rank {rank}: no inbound connection in time") from exc
    finally:
        listener.close()
    return Peer(inbound_sock), Peer(outbound_sock)


def barrier(inbound: Peer, outbound: Peer, rank: int) -> None:
    """A token makes one full lap of the ring; when rank 0 gets it back,
    every worker has reached this point."""
    if rank == 0:
        outbound.send_bytes(b"BARRIER")
    token = inbound.recv_bytes()
    if rank == 0:
        assert token == b"BARRIER", f"barrier token corrupted: {token!r}"
    else:
        outbound.send_bytes(token)
