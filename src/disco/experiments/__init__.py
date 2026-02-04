# src/disco/experiments/__init__.py
"""
Experiment API.

This package intentionally exposes a small public surface:
- Dataclasses representing experiment state
- Status enum
- Metastore-backed persistence + atomic update helper (ExperimentStore)
"""

from __future__ import annotations

from .experiment import (
    Assignment,
    Experiment,
    ExperimentStatus,
    Replication,
)
from .store import (
    ExperimentStore,
    Submission,
    experiment_path,
    EXPERIMENTS_ROOT,
    SUBMISSIONS_QUEUE,
)

__all__ = [
    "ExperimentStatus",
    "Assignment",
    "Replication",
    "Experiment",
    "ExperimentStore",
    "Submission",
    "experiment_path",
    "EXPERIMENTS_ROOT",
    "SUBMISSIONS_QUEUE",
]
