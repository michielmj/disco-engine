"""
disco.model

Model loading and specification utilities.

This subpackage provides:
- Pydantic-based parsing/validation of `model.yml` (ModelSpec)
- Loading models from:
  - installed Python packages
  - entry points (group: "disco.models") with a STRICT contract:
      entry point callable MUST return a package name string
  - a local dev folder (fallback)

The loaded runtime object is `Model` (validated spec + imported Node classes + attached ORM bundle).
"""

from __future__ import annotations

from .loader import (
    Model,
    ModelLoadError,
    discover_model_entrypoints,
    load_model,
    load_model_from_entrypoint,
    load_model_from_package,
    load_model_from_path,
)
from .orm import (
    ModelOrmError,
    ModelOrmLoadError,
    ModelOrmValidationError,
    OrmBundle,
    build_orm_bundle,
    db_install,
    db_validate,
)
from .spec import ModelSpec, NodeTypeSpec

__all__ = [
    # runtime
    "Model",
    # loading
    "ModelLoadError",
    "discover_model_entrypoints",
    "load_model",
    "load_model_from_entrypoint",
    "load_model_from_package",
    "load_model_from_path",
    # orm
    "OrmBundle",
    "ModelOrmError",
    "ModelOrmLoadError",
    "ModelOrmValidationError",
    "build_orm_bundle",
    "db_install",
    "db_validate",
    # spec
    "ModelSpec",
    "NodeTypeSpec",
]
