# graphdata-default-fill: `GraphData` offers `default_fill`

There is no requirement for this branch. It builds the owner's decision on
carry-over 26 (`docs/carry-overs.md`), 2026-11-27, verbatim: "default_fill
may be included as optional argument in the respective GraphData methods."
A parallel branch, `ci-macos-runner`, changes only `.github/workflows/`.

Before designing anything, read `ENGINEERING.md` Ch 7 (around line 2170, the
`default_fill` contract of the extract functions) and the README's "Filling
missing rows" bullet.

The brief's statements about the code are only as reliable as the code:
check each one, and record any that proves wrong as a DISCREPANCY.

## What this branch builds

1. **`default_fill` on the three `GraphData` methods.**
   `SimProcGraphData.outbound_edge_data` and `.inbound_edge_data`
   (`src/disco/graph_data.py:56,74`) and `GraphData.vertex_data` (line 194)
   take an optional keyword `default_fill`, default `None`, and pass it
   unchanged to `get_outbound_edge_data`, `get_inbound_edge_data` and
   `get_vertex_data`. A scalar and a per-column mapping both reach pandas as
   given.
2. **Docs follow.** Ch 7 (lines 2176-2177 say `GraphData` and
   `SimProcGraphData` do not forward it) and the README bullet ("`GraphData`
   does not offer `default_fill`: fill the frame it returns yourself") say
   that they now do.

Delete carry-over 26 from `docs/carry-overs.md`.

## Responsibility and cost

- **Whose job it is:** filling is pandas' (decision 0002); the extract
  functions already hand `default_fill` to `df.fillna` unchanged. disco's
  part here is only to pass the argument through: no checks, conversion or
  defaults of its own.
- **Upstream guidance read:** none new; pandas' `fillna` takes a scalar or a
  dict per column, as recorded on `dependency-fixes`.
- **Data path:** graph extraction, through `GraphData`. With
  `default_fill=None` each method must make exactly the call it makes today,
  so there is no cost to measure; no benchmark is asked. A non-`None` fill
  costs what pandas' `fillna` costs, which is the owner's choice.

## Where this is most likely to go wrong

1. **The argument does not reach pandas.** Pin the main path once per
   method: through each `GraphData` method, a per-column mapping on a
   mixed table gives the same frame as calling the extract function with it.
   Show each failing when the method drops the argument.
2. **The default changes today's call.** Without `default_fill`, each method
   passes `None` (or nothing), so missing rows stay missing, as today. The
   existing `GraphData` tests cover that path; keep them green.
3. **`inbound_edge_data` drifts from `outbound_edge_data`.** The two have
   different `columns` annotations today (one accepts `str`); don't change
   that here, but give `default_fill` the same name, position (keyword-only)
   and type on all three methods.

## Constraints

- No new dependency. No change to the extract functions.
- Your ports and worktree are in your launch prompt. `ci-macos-runner`
  works only on `.github/workflows/`.
- Graphblas cannot be imported on this container's Python 3.15 beta: use
  `uv run --python 3.14`, and say so in the Evidence.

## Done when

- Item 1 is pinned per method by a test shown failing (the prove-a-rule
  skill).
- The checks in CLAUDE.md § Commands pass, with tests on Linux.
- Somebody has used it: as a model author would, call
  `GraphData.vertex_data` with `default_fill={"count": 0, "name": ""}` and
  report the frame.
- Ch 7 and the README are current.
- The branch is pushed, with a Surfaced decisions commit whose Evidence
  lines give the test counts, each mutation and its result, and what using
  it showed.
