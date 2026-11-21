---
paths:
  - "tests/**"
---

# Writing tests

A test exists to fail when its rule breaks, so write it against the property
itself.

## A test needs a reason (the proportionate-effort decision)

Write a test only for:
- a requirement rule's main path;
- an incident that occurred;
- a rule on security, personal data or data loss.

Every other edge case goes into your report as a backlog item, not into a
test: a guard against an error that has not happened costs upkeep on every
change and catches nothing.
- One case per rule or table entry, not every combination of them.
- A data-loss proof covers data that cannot be recalculated, such as what
  was imported or entered by hand; not results the application derives again
  on its next run.
- Don't hard-code facts that change with every branch, such as the latest
  migration, a count or a list of registered items: derive them from the
  source, or every addition churns old tests.
- A proof repeated per instance (each migration's upgrade and downgrade, each
  endpoint's contract) goes through one shared helper, not a hand-built test
  each time. Write the helper when the second instance arrives.

## How to write one

- Assert the property, not a proxy for it. Laziness is proven by changing the
  underlying file and requiring the result to notice; memory use and timing
  are proxies.
- An invariant asserted on one path is a claim about that path only. Every
  path that must hold the invariant gets its own assertion.
- A memory bound is two assertions: a ratio across a 10× change in size, and
  a ceiling derived from the configuration. An absolute number is a guess.
- Timing is not an instrument at small scale: Windows timers have a 15 ms
  granularity.
- Seed fixtures from the shipped example content, so the suite also checks
  what ships. Don't build parallel minimal fixtures.
- Supply the tool, identity and encoding a test needs, and pass settings
  explicitly. A test that reads ambient host state passes on your machine and
  fails in CI. Don't set such state suite-wide either: that hides whether the
  production code sets it.
- Mark a test that takes more than a few seconds `@pytest.mark.slow`, and one
  that calls a real external service `@pytest.mark.live`.

When a test fails, fix the defect before you touch a bound: a threshold
relaxed first writes the defect into the acceptance criteria. Measure before
you optimise, and record each measurement with its configuration.

## In this repository

- Flat pytest functions only, never test classes: one style keeps fixtures
  and names predictable across the suite.
- Inject dependencies and use fakes rather than monkey-patching. Follow the
  existing `FakeKazooClient` and `FakeConnectionManager` for the Metastore
  when you need a new fake.
- Pass settings as `AppSettings(...)` keyword arguments; never rely on
  `get_settings()`, the environment, or a `config.toml` in the working
  directory.
