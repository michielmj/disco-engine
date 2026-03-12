# src/disco/partitioner/_protocol.py
from __future__ import annotations

from typing import Protocol

from disco.partitioning import Partitioning

NODE_TYPE = "node-type"


class Partitioner(Protocol):
    """
    A Partitioner produces a Partitioning for a given (Model, Graph) and target partition count.

    The returned Partitioning.num_partitions must be in [1, target_partition_count].
    """

    def partition(self, target_partition_count: int) -> Partitioning: ...
