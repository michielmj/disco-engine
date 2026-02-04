from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


class DiscoError(Exception):
    pass


class DiscoRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ExcDict:
    """
    Helper for merging exception dictionaries.

    Exception dicts are `dict[str, Any]`. When aggregating child exceptions:

    - For all keys except "description": values are collected into lists.
      * If the child value is a list/tuple, it is extended (flatten one level).
      * Otherwise it is appended.
    - For "description": child descriptions are joined into a single string
      separated by newlines ("\\n").

    Empty/None values are ignored. The merge is shallow (top-level keys only).
    """

    description_key: str = "description"

    def merge_many(self, excs: Iterable[Dict[str, Any] | None]) -> Optional[Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        desc_lines: List[str] = []

        for exc in excs:
            if not exc:
                continue

            # Special-case "description"
            if self.description_key in exc and exc[self.description_key] is not None:
                self._extend_description(desc_lines, exc[self.description_key])

            # All other keys become lists
            for k, v in exc.items():
                if k == self.description_key or v is None:
                    continue

                if isinstance(v, (list, tuple)):
                    if not v:
                        continue
                    merged.setdefault(k, [])
                    merged[k].extend(v)  # type: ignore[union-attr]
                else:
                    merged.setdefault(k, [])
                    merged[k].append(v)  # type: ignore[union-attr]

        if desc_lines:
            merged[self.description_key] = "\n".join(desc_lines)

        return merged or None

    @staticmethod
    def _extend_description(desc_lines: List[str], value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for x in value:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    desc_lines.append(s)
        else:
            s = str(value).strip()
            if s:
                desc_lines.append(s)


def _is_empty_exc(exc: Dict[str, Any] | None) -> bool:
    """
    Treat None, {}, {"description": ""}, {"description": None}, {"k": []} as empty.
    """
    if not exc:
        return True
    for k, v in exc.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        if isinstance(v, (list, tuple)) and len(v) == 0:
            continue
        # anything else is considered non-empty
        return False
    return True


def _canon_value(v: Any) -> Any:
    """
    Canonicalize values for equality:

    - Strings are stripped.
    - Lists/tuples are canonicalized element-wise and sorted when possible.
    - Dicts are canonicalized recursively (key-sorted).
    - Other values are kept as-is.
    """
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s
    if isinstance(v, dict):
        return tuple((str(k), _canon_value(val)) for k, val in sorted(v.items(), key=lambda kv: str(kv[0])))
    if isinstance(v, (list, tuple)):
        canon_items = [_canon_value(x) for x in v if x is not None]
        # drop empty strings
        canon_items = [x for x in canon_items if not (isinstance(x, str) and x == "")]
        # sort if comparable; else keep stable order by string repr
        try:
            return tuple(sorted(canon_items))  # type: ignore[type-var]
        except TypeError:
            return tuple(sorted(canon_items, key=lambda x: repr(x)))
    return v


def exc_equals(exc1: Dict[str, Any] | None, exc2: Dict[str, Any] | None) -> bool:
    """
    Equality helper for aggregated exception dictionaries.

    Semantics:
      - All "empty" exception representations are considered equal.
      - Otherwise compare canonicalized representations:
        * keys are compared in sorted order
        * strings are stripped
        * lists/tuples are treated as multisets (sorted) after canonicalization
        * dict values are canonicalized recursively
    """
    if _is_empty_exc(exc1) and _is_empty_exc(exc2):
        return True
    if _is_empty_exc(exc1) != _is_empty_exc(exc2):
        return False

    assert exc1 is not None and exc2 is not None

    c1 = tuple((str(k), _canon_value(v)) for k, v in sorted(exc1.items(), key=lambda kv: str(kv[0])))
    c2 = tuple((str(k), _canon_value(v)) for k, v in sorted(exc2.items(), key=lambda kv: str(kv[0])))
    return c1 == c2
