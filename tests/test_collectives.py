import pytest
import torch

from ddp.transport import Peer
from ring_helper import run_ring


def _check_allclose(results, key="values", tol=1e-4):
    for r in results:
        got = torch.tensor(r[key])
        exp = torch.tensor(r["expected"])
        assert torch.allclose(got, exp, atol=tol, rtol=tol), \
            f"max diff {(got - exp).abs().max()}"


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_allreduce_sum_matches_expected(world_size):
    # 1000 elements is not divisible by 3 -> exercises chunk padding.
    results = run_ring(world_size, "allreduce")
    assert len(results) == world_size
    _check_allclose(results)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_broadcast_from_rank_zero(world_size):
    results = run_ring(world_size, "broadcast")
    assert len(results) == world_size
    _check_allclose(results)


def test_tensor_roundtrip_over_socket():
    import socket
    import threading

    torch.manual_seed(7)
    original = torch.randn(64, 32)
    received = {}

    def server():
        srv = socket.socket()
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        received["port"] = srv.getsockname()[1]
        conn, _ = srv.accept()
        received["tensor"] = Peer(conn).recv_tensor()
        conn.close()
        srv.close()

    t = threading.Thread(target=server)
    t.start()
    while "port" not in received:
        pass
    cli = socket.socket()
    cli.connect(("127.0.0.1", received["port"]))
    Peer(cli).send_tensor(original)
    cli.close()
    t.join(timeout=10)
    assert torch.equal(received["tensor"], original)
