from __future__ import annotations

import argparse
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union, cast, get_args, get_origin

from pydantic import BaseModel


def _is_optional(tp: Any) -> tuple[bool, Any]:
    origin = get_origin(tp)
    if origin is Union:
        args = tuple(a for a in get_args(tp))
        if len(args) == 2 and type(None) in args:
            other = args[0] if args[1] is type(None) else args[1]
            return True, other
    return False, tp


def _is_list(tp: Any) -> tuple[bool, Any]:
    origin = get_origin(tp)
    if origin in (list, List):  # type: ignore[name-defined]
        args = get_args(tp)
        inner = args[0] if args else Any
        return True, inner
    return False, tp


def _is_literal(tp: Any) -> tuple[bool, Tuple[Any, ...]]:
    origin = get_origin(tp)
    if origin is not None and str(origin).endswith("Literal"):
        return True, get_args(tp)
    # Python 3.11: origin == typing.Literal
    if origin is getattr(__import__("typing"), "Literal", None):
        return True, get_args(tp)
    return False, ()


def _python_type_for_argparse(tp: Any) -> type:
    # We let Pydantic validate complex types (IPvAnyAddress, etc.) from strings.
    if tp in (str, int, float):
        return cast(type, tp)
    return str


def add_model_to_parser(parser: argparse.ArgumentParser, model: Type[BaseModel]) -> None:
    """
    Add arguments for all fields of a Pydantic v2 model to an argparse parser.
    Parsed values are intended to be passed to model.model_validate(vars(args)).
    """
    for name, field in model.model_fields.items():
        ann = field.annotation
        if ann is None:
            ann = Any

        required = field.is_required()

        # Help text from Field(description=...)
        help_text = field.description or ""

        # Decide argparse flag name
        flag = f"--{name.replace('_', '-')}"
        dest = name

        # Optional unwrap
        _, inner_ann = _is_optional(ann)

        # Bool flags
        if inner_ann is bool:
            # boolean optional action gives --flag / --no-flag
            default = field.default if field.default is not None else False
            parser.add_argument(
                flag,
                dest=dest,
                action=argparse.BooleanOptionalAction,
                default=default,
                help=help_text,
            )
            continue

        # Literals -> choices
        is_lit, choices = _is_literal(inner_ann)
        if is_lit:
            # argparse needs concrete strings typically; keep as-is and let pydantic validate
            default = None if required else field.default
            parser.add_argument(
                flag,
                dest=dest,
                choices=list(choices),
                default=default,
                required=required,
                help=help_text,
            )
            continue

        # Lists
        origin = get_origin(inner_ann)
        if origin is list:
            args = get_args(inner_ann)
            elem_ann = args[0] if args else str
            elem_type = _python_type_for_argparse(elem_ann)
            default = None if required else field.default
            parser.add_argument(
                flag,
                dest=dest,
                nargs="*",
                type=elem_type,
                default=default,
                required=required,
                help=help_text,
            )
            continue

        # Scalars / fallback to str
        arg_type = _python_type_for_argparse(inner_ann)
        default = None if required else field.default
        parser.add_argument(
            flag,
            dest=dest,
            type=arg_type,
            default=default,
            required=required,
            help=help_text,
        )
