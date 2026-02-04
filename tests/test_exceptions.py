from __future__ import annotations

from typing import Any, Dict, Optional

from disco.exceptions import ExcDict, exc_equals


def test_excdict_merge_many_returns_none_for_no_input() -> None:
    m = ExcDict()
    assert m.merge_many([]) is None
    assert m.merge_many([None, None]) is None
    assert m.merge_many([{}]) is None


def test_excdict_merges_description_with_newlines_and_flattens_lists() -> None:
    m = ExcDict()

    out = m.merge_many(
        [
            {"description": "one"},
            {"description": ["two", " three ", "", None]},
            {"description": None},
        ]
    )

    assert out is not None
    assert out["description"] == "one\ntwo\nthree"


def test_excdict_collects_other_keys_into_lists_and_extends_lists() -> None:
    m = ExcDict()

    out = m.merge_many(
        [
            {"code": 1, "tags": ["a", "b"]},
            {"code": 2, "tags": ["c"], "meta": {"x": 1}},
            {"code": None, "tags": [], "meta": None},
        ]
    )

    assert out is not None

    # scalars are appended
    assert out["code"] == [1, 2]

    # lists are extended
    assert out["tags"] == ["a", "b", "c"]

    # dict values are treated as scalar and appended
    assert out["meta"] == [{"x": 1}]


def test_excdict_includes_description_alongside_other_keys() -> None:
    m = ExcDict()

    out = m.merge_many(
        [
            {"description": "a", "code": 1},
            {"description": "b", "code": 2},
        ]
    )

    assert out == {"code": [1, 2], "description": "a\nb"}


def test_excdict_ignores_empty_and_none_values() -> None:
    m = ExcDict()

    out = m.merge_many(
        [
            {"description": "", "code": None, "tags": []},
            None,
            {},
            {"description": "ok"},
        ]
    )

    assert out == {"description": "ok"}


def test_exc_equals_empty_variants() -> None:
    assert exc_equals(None, None)
    assert exc_equals(None, {})
    assert exc_equals({}, {"description": ""})
    assert exc_equals({"k": []}, None)


def test_exc_equals_description_whitespace_ignored() -> None:
    assert exc_equals({"description": "a"}, {"description": " a "})


def test_exc_equals_list_order_ignored() -> None:
    assert exc_equals({"codes": [2, 1]}, {"codes": [1, 2]})