from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module

# noinspection PyProtectedMember
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

from disco.database import DbHandle
from disco.exceptions import DiscoRuntimeError
from .orm import ModelOrmError, OrmBundle, build_orm_bundle
from .spec import ModelSpec


class ModelLoadError(DiscoRuntimeError):
    """Raised when a model cannot be discovered, loaded, imported, or validated."""


@dataclass(frozen=True, slots=True)
class Model:
    """
    Runtime Model: validated spec + imported Node classes + attached ORM bundle.

    - Defines types only (simprocs, node types).
    - Scenario defines instances (vertices/edges + node/edge data).
    """

    spec: ModelSpec
    node_classes: Dict[str, type]
    orm: OrmBundle

    def node_class(self, node_type: str) -> type:
        try:
            return self.node_classes[node_type]
        except KeyError as e:
            raise KeyError(f"Unknown node type '{node_type}'. Known: {sorted(self.node_classes)}") from e

    def node_factory(self, node_type: str, *args, **kwargs):
        """
        Return a constructed Node instance (not initialized with runtime resources).

        NodeController will call node.initialize(experiment=..., replication=..., resources=..., params=...).
        """
        cls = self.node_class(node_type)
        try:
            node = cls(*args, **kwargs)  # must be a lightweight, no-arg ctor by convention
        except TypeError as e:
            raise ModelLoadError(
                f"Failed to construct node type '{node_type}' from class "
                f"'{cls.__module__}.{cls.__qualname__}'. "
                "Node classes must support a no-arg __init__()."
            ) from e
        return node


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    """
    Temporarily prepend `path` to sys.path. Used only for raw dev folder mode.
    """
    import sys

    p = str(path)
    old = list(sys.path)
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path[:] = old


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise ModelLoadError(f"model.yml not found at: {path}") from e
    except OSError as e:
        raise ModelLoadError(f"Failed to read model.yml at: {path}: {e}") from e


def _parse_model_yml(text: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)  # type: ignore[attr-defined]
    except Exception as e:
        raise ModelLoadError(f"Failed to parse model.yml (YAML error): {e}") from e
    if not isinstance(data, dict):
        raise ModelLoadError("model.yml must parse to a mapping/dict at the root")
    return data


def _parse_python_class_ref(ref: str) -> tuple[str, str]:
    """
    Accepts either:
      - "pkg.module:ClassName"
      - "pkg.module.ClassName"

    Returns (module_path, class_name).
    """
    r = ref.strip()
    if ":" in r:
        mod, cls = r.split(":", 1)
        mod, cls = mod.strip(), cls.strip()
    else:
        if "." not in r:
            raise ModelLoadError(
                f"Invalid class reference '{ref}'. Expected 'pkg.module:ClassName' or 'pkg.module.ClassName'."
            )
        mod, cls = r.rsplit(".", 1)
        mod, cls = mod.strip(), cls.strip()

    if not mod or not cls:
        raise ModelLoadError(
            f"Invalid class reference '{ref}'. Expected 'pkg.module:ClassName' or 'pkg.module.ClassName'."
        )
    return mod, cls


def _import_node_class(ref: str) -> type:
    mod_path, cls_name = _parse_python_class_ref(ref)
    try:
        mod = import_module(mod_path)
    except Exception as e:
        raise ModelLoadError(f"Failed to import module '{mod_path}' for node class '{ref}': {e}") from e

    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise ModelLoadError(f"Module '{mod_path}' has no attribute '{cls_name}' (from '{ref}')") from e

    if not isinstance(cls, type):
        raise ModelLoadError(f"'{ref}' did not resolve to a class (got {type(cls)!r})")
    return cls


def _validate_node_subclass(node_cls: type) -> None:
    """
    Validate class derives from disco.node.Node.
    (Import locally to avoid circular import.)
    """
    from disco.node import Node  # local import avoids circular import at import-time

    if not issubclass(node_cls, Node):
        raise ModelLoadError(
            f"Node class '{node_cls.__module__}.{node_cls.__qualname__}' must subclass disco.node.Node"
        )


def _build_node_registry(spec: ModelSpec) -> Dict[str, type]:
    out: Dict[str, type] = {}
    for node_type_name, nts in spec.node_types.items():
        node_cls = _import_node_class(nts.python_class)
        _validate_node_subclass(node_cls)
        out[node_type_name] = node_cls
    return out


def _attach_orm(
    *,
    spec: ModelSpec,
    db: Optional[DbHandle],
    schema: Optional[str],
) -> OrmBundle:
    """
    Attach ORM bundle (tables resolved from provider metadata or reflected from DB).

    No DB writes occur here. For deliberate actions use:
      - disco.model.orm.db_validate(model.orm, db)
      - disco.model.orm.db_install(model.orm, db)
    """
    try:
        return build_orm_bundle(spec=spec, db=db, schema=schema)
    except ModelOrmError as e:
        raise ModelLoadError(
            f"Failed to attach ORM for model '{getattr(spec, 'name', None) or '<unnamed>'}': {e}"
        ) from e


# -----------------------------
# Package-based loading
# -----------------------------


def load_model_from_package(
    package: str,
    *,
    db: Optional[DbHandle] = None,
    schema: Optional[str] = None,
    model_yml: str = "model.yml",
) -> Model:
    """
    Load a model from an installed Python package and attach ORM bundle.

    Assumes `model.yml` is shipped inside the package.
    Works for wheels and for `pip install -e .`.
    """
    try:
        import_module(package)
    except Exception as e:
        raise ModelLoadError(f"Failed to import model package '{package}': {e}") from e

    try:
        from importlib import resources
        text = (resources.files(package) / model_yml).read_text(encoding="utf-8")
    except Exception as e:
        raise ModelLoadError(
            f"Failed to read '{model_yml}' from package '{package}'. "
            "Ensure it is included as package data. Error: "
            f"{e}"
        ) from e

    data = _parse_model_yml(text)
    spec = ModelSpec.model_validate(data)
    node_classes = _build_node_registry(spec)
    orm = _attach_orm(spec=spec, db=db, schema=schema)
    return Model(spec=spec, node_classes=node_classes, orm=orm)


# -----------------------------
# Raw dev-folder loading (fallback)
# -----------------------------


def load_model_from_path(
    path: str | Path,
    *,
    db: Optional[DbHandle] = None,
    schema: Optional[str] = None,
    model_yml: str = "model.yml",
    import_root: Optional[str | Path] = None,
) -> Model:
    """
    Load a model from a folder on disk (dev fallback) and attach ORM bundle.

    - Expects `<path>/<model_yml>` to exist.
    - Temporarily adds `import_root` to sys.path to allow imports of node classes and ORM provider.

    Recommended dev flow is still: `pip install -e <path>` and then use load_model_from_package().
    """
    path = Path(path).resolve()
    yml_path = path / model_yml
    text = _read_text_file(yml_path)
    data = _parse_model_yml(text)
    spec = ModelSpec.model_validate(data)

    root = Path(import_root).resolve() if import_root is not None else path

    with _temporary_sys_path(root):
        node_classes = _build_node_registry(spec)
        orm = _attach_orm(spec=spec, db=db, schema=schema)

    return Model(spec=spec, node_classes=node_classes, orm=orm)


# -----------------------------
# Entry points (STRICT CONTRACT)
# -----------------------------


def discover_model_entrypoints(group: str = "disco.models") -> Dict[str, EntryPoint]:
    eps = entry_points(group=group)
    return {ep.name: ep for ep in eps}


def load_model_from_entrypoint(
    name: str,
    *,
    db: Optional[DbHandle] = None,
    schema: Optional[str] = None,
    group: str = "disco.models",
    model_yml: str = "model.yml",
) -> Model:
    """
    STRICT CONTRACT:

    The entry point must resolve to a callable that returns a *package name* (str).

    That package must be importable on the worker, and must contain `model.yml`
    (or the configured model_yml resource), and referenced Node classes.
    """
    eps = discover_model_entrypoints(group=group)
    if name not in eps:
        available = ", ".join(sorted(eps.keys())) or "(none)"
        raise ModelLoadError(f"No model entry point '{name}' in group '{group}'. Available: {available}")

    ep = eps[name]

    try:
        target = ep.load()
    except Exception as e:
        raise ModelLoadError(f"Failed to load entry point '{name}' ({ep.value}): {e}") from e

    if not callable(target):
        raise ModelLoadError(
            f"Entry point '{name}' did not resolve to a callable (got {type(target)!r}). "
            "It must resolve to a callable that returns a package name string."
        )

    try:
        result = target()
    except Exception as e:
        raise ModelLoadError(f"Entry point '{name}' callable raised: {e}") from e

    if not isinstance(result, str):
        raise ModelLoadError(
            f"Entry point '{name}' must return a package name string (str), but returned {type(result)!r}."
        )

    package = result.strip()
    if not package:
        raise ModelLoadError(f"Entry point '{name}' returned an empty package name string.")

    return load_model_from_package(package, db=db, schema=schema, model_yml=model_yml)


# -----------------------------
# Unified loader facade
# -----------------------------


def load_model(
    *,
    plugin: Optional[str] = None,
    package: Optional[str] = None,
    path: Optional[str | Path] = None,
    db: Optional[DbHandle] = None,
    schema: Optional[str] = None,
    model_yml: str = "model.yml",
    dev_import_root: Optional[str | Path] = None,
) -> Model:
    """
    Unified entry for loading a model and attaching ORM bundle.

    Exactly one of plugin/package/path must be provided.

    Notes:
    - If model.yml does NOT specify an ORM provider (spec.orm is None), db must be provided (reflection mode).
    - No DB writes occur here. Use db_validate/db_install explicitly.
    """
    provided = [plugin is not None, package is not None, path is not None]
    if sum(provided) != 1:
        raise ModelLoadError("Exactly one of plugin, package, or path must be specified")

    if plugin is not None:
        return load_model_from_entrypoint(plugin, db=db, schema=schema, model_yml=model_yml)

    if package is not None:
        return load_model_from_package(package, db=db, schema=schema, model_yml=model_yml)

    assert path is not None
    return load_model_from_path(path, db=db, schema=schema, model_yml=model_yml, import_root=dev_import_root)
