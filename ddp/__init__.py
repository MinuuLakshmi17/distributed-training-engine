"""Data-parallel SGD over a TCP ring."""
from .train import launch, worker_main

__all__ = ["launch", "worker_main"]
