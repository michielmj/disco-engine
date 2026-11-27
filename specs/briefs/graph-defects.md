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

## Round 1 (2026-11-25): float weights, full columns, and a test that can fail

The review of d39eb65 found three FAILs. The cycle check, the mask cleanup
code and the `graph.md` facts were confirmed correct. A re-review, scoped to
this round, follows: fixes 1 and 2 change behaviour.

<finding id="1">
src/disco/partitioning.py:455: `.dup(dtype=INT64)` truncates float weights,
so a layer edge weighted in (-1, 1) (for example 0.5) becomes 0. The two
plus_times products then give 0, `select("!=", 0)` drops the link, and the
node loses that predecessor or successor. Edge weights are `Float NOT NULL`
(graph/schema.py:44), so every graph loaded from the database has FP64
layers. On the two-paths-join graph with FP64 0.5: branch `n0 successors:
{}`, staging `{'s0': {('n1','s0')}}`. FP64 2.5 and INT64 1 are the same on
both. No test covers non-BOOL weights.
</finding>

**Remedy (prescribed):** the topology depends only on which edges exist, so
compute it on the pattern: cast BOOL layers only, or take the pattern of
every layer as INT64 ones (`M.apply(gb.unary.one).new(dtype=gb.dtypes.INT64)`
or equivalent). Pin it with an FP64 layer of weight 0.5, giving the same
topology as staging, shown failing on d39eb65's code. Narrow the new
`graph.md` advice so it no longer reads as "cast any layer to INT64": the
cast is for counting on BOOL; a float layer must not be truncated.

<finding id="2">
src/disco/graph/extract.py:63-70 (`_fill_missing`): a column with no missing
values is still cast to `object` when the fill does not match its type. On
the real `get_vertex_data` path, with SQLite and a table where every value
is present: `default_fill=0`: branch gives `key: object, name: object`,
staging gives `key: str, name: str`. `default_fill=""`: branch gives
`count: object`, staging gives `count: int64`. The new test passes only
because each of its text columns has a missing value.
</finding>

**Remedy (prescribed):** touch a column only when it has a missing value
(`col.isna().any()`). Pin it with full columns of each kind (int, float,
`str`) under fills `0` and `""`, comparing dtypes and values with staging,
shown failing on d39eb65's code. While there, cover the case the review
also found: a `datetime64` column with a numeric fill still emits
`Pandas4Warning`. Brief item 3 asks for none on the extraction path, so
handle it with the same rule (same result as today, no warning).

<finding id="3">
tests/graph/test_graph_mask.py::test_written_masks_are_stamped_with_naive_utc:
the test is named for the rule "stored values stay naive UTC", but its
`s.tzinfo is None` check cannot fail, because SQLite drops tzinfo. This can
be checked without PostgreSQL.
</finding>

**Remedy (prescribed):** assert on what the code hands the database, not on
what SQLite gives back: capture the bound parameters with a SQLAlchemy
`before_cursor_execute` listener (or assert `graph_mask._utcnow().tzinfo is
None` and that the writes use it). Show it failing with an aware
`datetime.now(UTC)`. Then the PostgreSQL item in your Evidence is no longer
needed for this rule.

**Also in this round:**
- `ENGINEERING.md` Ch 7's `vertex_masks` section (around line 1632): state
  that `created_at` and `updated_at` hold naive UTC. That is the contract
  the cleanup relies on.
- Not in this round, and recorded by the architect as carry-overs: the
  `utcnow()` call in `src/disco/graph/db.py:252`, the empty graph (n=0)
  reported as cyclic, and `_fill_missing` on duplicate column names. Don't
  add them to `docs/carry-overs.md` here.

Fresh Evidence lines on the fixed head: the round's tests and mutations, the
full suite (with `uv run --python 3.14`), and the checks.

## Round 2 (2026-11-25): stored zeros, and datetime under an empty fill

The scoped re-review of 746a118 found two FAILs, both changes from
staging's results that round 1 introduced. It confirmed the rest of round 1.
A scoped re-review follows: remedy 2 is described.

<finding id="1">
src/disco/partitioning.py:456 `.apply(gb.unary.one)` turns an explicitly
stored zero weight into an edge. On staging, a 0 weight summed to 0 and
`select("!=", 0)` dropped the link. Edge weights are `Float NOT NULL`, and
`_load_edge_layers` (graph/db.py:325-350) stores a 0.0 row as an entry, so a
DB edge with weight 0 now adds a predecessor/successor link that staging
did not. FP64 0.0 gives branch `n0 successors {'s0': {('n1','s0')}}` and
staging `{}`; INT64 0 gives the same split. The codebase's own convention
drops zeros: `helpers/gb.py:81` uses `matrix.select("!=", 0).apply(unary.one)`
("keep only real edges").
</finding>

**Remedy (prescribed):** `M.select("!=", 0).apply(gb.unary.one).new(dtype=gb.dtypes.INT64)`,
the convention of `helpers/gb.py:81`. Add a stored zero (FP64 0.0) to the
parametrized topology test, expecting staging's result, shown failing on
746a118. Add "drop stored zeros first" to the `.claude/rules/graph.md`
advice for float layers.

The architect decides one consequence: weights that cancelled (+1 and -1
into the same node pair) gave no link on staging and now give one. An edge
exists on both paths, so the link is right and staging's result was a
numeric artefact. Record it as DECIDED; no test is needed for it.

<finding id="2">
src/disco/graph/extract.py:70-71 now treats a datetime64 column as
compatible only with a datetime fill. A datetime64 column (naive or
tz-aware) that has a missing value, under `default_fill=""`, now becomes
`object` holding `""`. On staging it stayed `datetime64[us]` holding `NaT`,
with no warning. Comparing staging's `df.fillna` with the branch's
`_fill_missing` (pandas 3.1.1): `'' dt miss staging=('datetime64[us]',
[Timestamp, NaT]) | branch=('object', [Timestamp, ''])`, and the same for a
tz-aware column. `test_default_fill_of_a_datetime_column_warns_nothing` only
tests fill 0. At d39eb65 datetime was `compatible = True`, which matched
staging for `""`.
</finding>

**Remedy (described):** a datetime64 column with a missing value gives
staging's result under each fill a model author uses (`0` and `""` at
least; naive and tz-aware), with no `Pandas4Warning`. Pin fill `""` beside
the existing fill-0 test, shown failing on 746a118. If pandas offers no way
to give staging's result for some fill without the warning, stop and record
it as ESCALATED with the case, rather than choose a new result.

**Also in this round:**
- `ENGINEERING.md` Ch 7 (around line 1639) and the `_utcnow` docstring say
  `graph_vertex_masks` has `created_at` and `updated_at`. It has only
  `updated_at` (`graph/schema.py:85`); `created_at` belongs to
  `graph_scenarios`. The error came from this brief's round 1 (the
  architect's), not from you. Correct both to the schema.
- Not in this round, recorded by the architect as backlog: `Pandas4Warning`
  for a timedelta64 column under a numeric fill and for a tz-aware column
  under a naive Timestamp fill (both on staging too), and nullable `Int64`
  under fill `""`.

Fresh Evidence lines on the fixed head: the round's tests and mutations, the
full suite (with `uv run --python 3.14`), and the checks.
