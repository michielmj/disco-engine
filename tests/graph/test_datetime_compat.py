import ast
from pathlib import Path


def test_graph_modules_avoid_deprecated_utcnow() -> None:
    root = Path(__file__).parents[2]
    paths = [
        root / "src" / "disco" / "graph" / "db.py",
        root / "src" / "disco" / "graph" / "graph_mask.py",
    ]

    for path in paths:
        tree = ast.parse(path.read_text())
        utcnow_lines = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "utcnow"
        ]
        assert utcnow_lines == []
