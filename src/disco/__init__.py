try:
    from ._version import version as __version__  # populated by setuptools-scm
except ModuleNotFoundError:
    __version__ = "0.0.0"

from .node import Node

__all__ = ["__version__", "Node"]

