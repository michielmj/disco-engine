# graph-defects: the graph and graphblas defects of wave 1

There is no requirement for this branch: it fixes carry-overs 8, 9, 11, 12
and 14 in `docs/carry-overs.md`, which the owner made priority 1 on
2026-11-25. Read those five rows first; each names the file and line. The
other wave 1 branches run in parallel: `tooling-fixes` (the guard hook and
`CLAUDE.md`) and later `testrun-params` (`testrun.py` and its tests).

Before designing anything, read `ENGINEERING.md` Ch 7 (graph) and Ch 18
(partitioner), and `.claude/rules/graph.md`.

The brief's statements about the code are only as reliable as the code:
check each one, and record any that proves wrong as a DISCREPANCY.

## What this branch builds

1. **The partitioner handles BOOL weights** (carry-over 8). A graph whose
   layer has BOOL weights partitions without error, with the same result as
   the same graph with INT64 weights of 1. Today `gb.op.plus_times(incidence
   @ M)` at `src/disco/partitioning.py:456` raises `KeyError`.
2. **The cycle check is right on acyclic graphs** (carry-over 9).
   `matrix_has_cycle` returns False for the chain 0→1→2 and True for a real
   cycle and a self-loop. The cause recorded is that `_longest_path_length`
   (`src/disco/helpers/gb.py`) runs n-1 iterations and never sees
   convergence; confirm it. `longest_path_length` must still return the
   right lengths, since `graph.core` (line 172) and others rely on it.
3. **Graph extraction is ready for pandas 4** (carry-over 11). The three
   `df.fillna(default_fill)` calls in `src/disco/graph/extract.py` (lines
   148, 369, 483) emit no `Pandas4Warning`, and return the same frames as
   today, `str` columns included.
4. **No deprecated `utcnow`** (carry-over 12). `src/disco/graph/graph_mask.py`
   (lines 80, 95, 122) no longer calls `datetime.utcnow()`. PR #44 was
   closed unmerged; don't reuse it without checking it against item 4 below.
5. **`graph.md` says what graphblas does** (carry-over 14). Correct the
   wording of `.claude/rules/graph.md` lines 15–17 to the facts in
   carry-over 14, which hold on graphblas 2025.2 and 2026.11. The owner
   approved this ("ok", 2026-11-25). Use the corrected rule for item 1.

Delete the five rows from `docs/carry-overs.md` in this branch.

## Where this is most likely to go wrong

1. **Mask cleanup deletes live masks.** `graph_mask.py:80` deletes masks
   whose `updated_at` is older than a cutoff, and `updated_at` and
   `created_at` are naive `DateTime` columns (`graph/schema.py:24,85`)
   holding UTC. A timezone-aware replacement for `utcnow()` can shift or
   break that comparison against rows already in the database: that is data
   loss. Prove it on the database the tests use: a mask written before the
   change, with an `updated_at` inside the age limit, survives cleanup; one
   older than the limit is deleted. Stored values stay naive UTC, so
   existing databases need no migration.
2. **BOOL is OR, not count.** Under BOOL, `plus` is logical OR, so a fix
   that only avoids the `KeyError` can give the wrong partitioning. Pin
   item 1 with a graph where two paths join, so that OR and addition differ,
   and compare against INT64 weights.
3. **The cycle fix breaks the lengths.** Pin item 2 on the chain, on a
   cycle, on a self-loop, and on the longest-path lengths of a DAG with two
   branches of different length. `_longest_path_length` is `lru_cache`d
   on a pickled matrix; don't let one test's result leak into another.
4. **The `fillna` fix changes values.** Make the extraction tests treat
   `Pandas4Warning` as an error for these paths, and show that the frames
   are equal before and after, including a column of all-missing text.

## Constraints

- No new dependency, and no change to the graph's file or database format.
- `disco.graph.core` stays database-agnostic (Ch 7, `.importlinter`).
- Your ports and worktree are in your launch prompt. `tooling-fixes`
  works on `.claude/hooks/` and `CLAUDE.md`, `testrun-params` on
  `src/disco/testrun.py` and its tests: stay out of them.

## Done when

- Each item above is pinned by a test that fails when it is broken, shown
  failing (the prove-a-rule skill).
- The checks in CLAUDE.md § Commands pass, with tests on Linux.
- Somebody has used it:
  1. Partition the example model's graph, or a small graph you build as a
     model author would, with a BOOL layer, and report the result.
  2. Run the extraction path a model author uses with
     `-W error::pandas.errors.Pandas4Warning` and report that it is clean.

  What needs the owner is recorded as "not checked, for the owner".
- `ENGINEERING.md` Ch 7 and Ch 18 are current for any contract this
  changed (the sync-spec skill).
- The branch is pushed, with a Surfaced decisions commit whose Evidence
  lines give the test counts, each mutation and its result, and what using
  it showed.
