# dependency-fixes: let pandas, scipy and graphblas do their own work

There is no requirement for this branch. It fixes the two FAILs, and one
observation, of the full re-review of release PR #56 under decision 0002
(`docs/decisions/0002-dont-do-the-dependencies-work.md`), which the owner
directed on 2026-11-26. Read decision 0002 and `.claude/rules/dependencies.md`
first. Release PR #56 is `staging` itself, so this branch lands into
`staging` and #56 picks it up.

Before designing anything, read `ENGINEERING.md` Ch 7 (graph: extraction,
validation) and Ch 18 (partitioner), and `.claude/rules/graph.md`.

The brief's statements about the code are only as reliable as the code:
check each one, and record any that proves wrong as a DISCREPANCY.

## What this branch builds

1. **`default_fill` goes straight to pandas.** The owner chose option A
   ("A", 2026-11-26) when asked: "hand it straight to pandas, as `main` did.
   `default_fill` may also be a per-column mapping, such as `{"count": 0,
   "name": ""}`, which pandas already accepts. A value that doesn't fit gives
   pandas' warning today and an error under pandas 4. The README tells model
   authors to use a mapping for mixed tables. disco adds no code of its own,
   and there is no cost." So `get_vertex_data`, `get_outbound_edge_data` and
   `get_inbound_edge_data` (`src/disco/graph/extract.py`) call
   `df.fillna(default_fill)` once; `_fill_missing` and `_is_datetime_fill`
   are deleted, with the tests that pin their re-implemented results. A
   per-column mapping reaches pandas unchanged, through `GraphData` too.
   Ch 7 and the README say what a model author passes and what pandas does
   with a value that does not fit.
2. **The cycle check calls scipy.** `matrix_has_cycle`
   (`src/disco/helpers/gb.py`) answers with
   `scipy.sparse.csgraph.connected_components(directed=True,
   connection="strong")` on the layer's real edges: a directed cycle exists
   iff there are fewer strong components than vertices, or a vertex has a
   self-loop. Graph validation (`graph/core.py:172`) runs it on every layer.
   `longest_path_length` stays as it is (it is public in `disco.helpers`),
   but `matrix_has_cycle` no longer calls it.
3. **The partitioner's topology uses graphblas' structural idiom.** In
   `src/disco/partitioning.py` (around line 457), compute the topology on the
   layer's real edges with the `any_pair` semiring (`gb.semiring.any_pair`),
   which ignores values, instead of casting each layer to INT64 ones. Correct
   the advice in `.claude/rules/graph.md` (around line 21) to the same idiom.

Delete carry-overs 15 (empty graph reported cyclic: it must not be, after
item 2), 17 and 18 (gaps of `_fill_missing`, gone with it) from
`docs/carry-overs.md`.

## Responsibility and cost

- **Whose job it is:**
  - filling missing values, and deciding which fill suits which dtype:
    pandas. disco's part is the contract of `default_fill`, now decided by
    the owner (item 1).
  - detecting a directed cycle: scipy (`scipy.sparse.csgraph`), a runtime
    dependency (`pyproject.toml:32`, `scipy>=1.13`); disco already converts
    graphblas matrices to scipy with `gb.io.to_scipy_sparse`
    (`src/disco/partitioner/_spectral.py:203`). graphblas-algorithms has no
    cycle or DAG check (the re-review found only descendants and ancestors).
  - multiplying on a matrix's structure: graphblas (`any_pair`).
  - disco: which edges count (stored zeros are not edges, as
    `helpers/gb.py` and staging's partitioner treat them), and the
    contracts in Ch 7 and Ch 18.
- **Upstream guidance read:**
  - pandas 3.1.1's own deprecation, `pandas/core/internals/blocks.py:1457`
    ("GH#45153 fillna with incompatible value requiring any dtype casting is
    deprecated"): "In a future version, calling fillna with an incompatible
    value will raise. Explicitly cast to a common dtype before filling."
    pandas' `fillna` docstring: `value : scalar, dict, Series, or DataFrame`
    ... "dict ... of values specifying which value to use for each ...
    column. Values not in the dict ... will not be filled." The pandas web
    pages (release notes, pandas-dev/pandas#64959) returned 403 to the
    reviewer; read them if you can reach them, and record it either way.
  - scipy: the `connected_components` documentation for `directed`,
    `connection="strong"`, and how it treats explicitly stored zeros in a
    sparse matrix. Read it; do not assume.
- **Data path:** all three items are on data paths (extraction, graph
  validation, partitioning). Benchmark each before (staging) and after, and
  give the numbers with their configuration in your Evidence:
  - `_fill_missing` vs `df.fillna`: a 1M-row frame with and without missing
    values, and 2,000 frames of 10 rows (the per-node path). Re-review
    figures for reference: 3.2x and 4.8x slower than `fillna`.
  - `matrix_has_cycle`: a 100k-vertex chain, a 20k-vertex graph with a
    cycle, a random DAG of 100k vertices and 500k edges. Re-review figures on
    staging: 107 s, 19.4 s, 0.25 s; scipy: 0.9 ms, 1.0 ms, 19 ms.
  - topology: 1M vertices, 5 layers of 5M edges, 2,000 nodes. Re-review
    figures: staging about 0.8-1.0 s, `any_pair` 597 ms for the same
    14,253,408 pairs.
  No slowdown is approved.

## Where this is most likely to go wrong

1. **Stored zeros become edges, or self-loops are missed.** Staging drops
   stored zeros before both checks (`select("!=", 0)`), and scipy's
   csgraph may treat an explicitly stored zero as an edge. Pin: a layer
   whose only cycle runs through a stored zero is acyclic; a self-loop is a
   cycle; the chain 0→1→2 is acyclic; an empty graph is acyclic; a cycle
   with no source is a cycle. The partitioner's topology on a stored zero
   stays as staging's (`test_a_stored_zero_weight_is_not_an_edge` exists).
2. **Topology changes.** `any_pair` must give the same predecessor and
   successor sets as staging for BOOL, INT64 and FP64 layers, including
   negative and fractional weights (the parametrized topology tests exist;
   keep them green). The cancelling +1/-1 case gives a link, as decided on
   `graph-defects`.
3. **`default_fill` loses a case model authors use.** Pin the main path of
   item 1 through the public functions: a scalar fill on a numeric table,
   and a per-column mapping on a mixed table, each giving pandas' own
   result. Do not pin pandas' coercion rules or its warning in disco's
   tests: they are pandas' behaviour, and change with it.
4. **A test still locks the workaround in.** The re-review found that
   `test_default_fill_of_text_columns_warns_nothing_and_keeps_the_values`
   and `test_default_fill_of_a_datetime_column_warns_nothing` fail under
   plain `fillna`, and that
   `test_default_fill_of_a_datetime_column_with_an_empty_string_keeps_the_dtype`
   calls the private function. Remove or rewrite each, and say which.

## Constraints

- No new dependency: scipy and graphblas are already runtime dependencies.
- `disco.graph.core` stays database-agnostic; the layering contracts hold.
- Your ports and worktree are in your launch prompt. No other branch runs
  in parallel.
- Graphblas cannot be imported on this container's Python 3.15 beta: use
  `uv run --python 3.14`, and say so in the Evidence.

## Done when

- Items 1-3 are pinned by tests that fail when broken, shown failing (the
  prove-a-rule skill), on main paths and on the stored-zero data rule.
- The checks in CLAUDE.md § Commands pass, with tests on Linux.
- The benchmarks above, before and after, are in the Evidence; any
  slowdown is ESCALATED, not explained.
- Somebody has used it: as a model author would, load a graph whose layer
  has a cycle and one without (validation reports each correctly), and call
  `get_vertex_data` with a per-column mapping; report what you saw.
- `ENGINEERING.md` Ch 7 and Ch 18 and the README are current (the sync-spec
  skill).
- The branch is pushed, with a Surfaced decisions commit whose Evidence
  lines give the test counts, each mutation and its result, the benchmarks,
  and what using it showed.
