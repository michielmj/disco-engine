# src/disco/partitioner/__init__.py
"""
disco.partitioner
=================

Partitioner implementations for the disco engine.

A *Partitioner* assigns graph vertices to simulation-node partitions so that
compute load is balanced and cross-partition event traffic is minimised.

Public API
----------
- ``Partitioner``                   : structural Protocol (typing only).
- ``NODE_TYPE``                     : label-type name used to tag node-type labels.
- ``SimplePartitioner``             : always produces a single partition.
- ``SpectralClusteringPartitioner`` : spectral-clustering-based multi-partition splitter.
"""
from __future__ import annotations

from ._protocol import NODE_TYPE, Partitioner
from ._simple import SimplePartitioner
from ._spectral import SpectralClusteringPartitioner

__all__ = [
    "NODE_TYPE",
    "Partitioner",
    "SimplePartitioner",
    "SpectralClusteringPartitioner",
]
