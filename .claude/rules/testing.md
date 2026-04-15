# Testing conventions

- Flat pytest functions only. No test classes, ever.
- Full coverage is expected with every change. Tests must stay in sync with
  behavior — if behavior changes, tests change in the same task.
- Use dependency injection and fakes rather than monkey-patching where possible.
  The project already has FakeKazooClient / FakeConnectionManager for Metastore
  testing — follow the same pattern for new fakes.
- Run: `pytest`
- Type check: `mypy --strict src/disco`