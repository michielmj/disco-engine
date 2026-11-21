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
    EXPERIMENTS_ROOT,
    SUBMISSIONS_QUEUE,
    ExperimentStore,
    Submission,
    experiment_path,
)

__all__ = [
    "EXPERIMENTS_ROOT",
    "SUBMISSIONS_QUEUE",
    "Assignment",
    "Experiment",
    "ExperimentStatus",
    "ExperimentStore",
    "Replication",
    "Submission",
    "experiment_path",
]
