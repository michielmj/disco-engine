# 01: Restructuring: one public interface, a layered package, no technical debt in the layout

**Status: draft for the product owner's approval, 2026-11-27, revision 1.**
- **Next:** a review of this draft, then the owner's answers to any
  questions it raises, then the owner's approval. Nothing is briefed until
  then.
- **Once approved,** this document wins wherever it and a brief disagree.

## The requirement, in the product owner's words

2026-11-25, answering what the restructuring (carry-over 1) is for:

> The goal is removal of technical debt as a first step towards a first stable release.

Carry-over 1, as recorded on 2026-11-21: "Restructure the package layout.
Breaking changes are allowed (owner, 2026-11-21)."

2026-11-27, starting this requirement: "restructuring: removal of technical
debt as a first step towards a first stable release (carry-overs 1, 3, 6, 7;
mypy strict, item 2, follows)"

## How this was settled

**The owner's answers, verbatim.** Round 1, 2026-11-27, answered "A, A, A, A":

| Round | Question | Answer |
|---|---|---|
| 1 | Where do model authors import disco from after the restructuring? A: one documented public interface, for example `from disco import Node, Event, TestRun, Experiment, ...`, plus a few named subpackages such as `disco.graph`; everything else is internal and may move freely. B: every import path that works today keeps working. | "A" |
| 1 | What happens to old import paths? A: they stop working in the release that restructures; the README gets an old-to-new table. B: they keep working for one release with a deprecation warning, then go. | "A" |
| 1 | Do operators see any change? A: no: the commands (`disco`, `disco-server`), the `config.toml` keys and the environment variable names stay exactly as they are. B: they may change too, documented in the README. | "A" |
| 1 | Do the two slow data paths (carry-overs 23 and 24) join this requirement? A: no, they get their own small requirement after this one, so the restructuring changes no behaviour. B: yes. | "A" |

Round 2, 2026-11-27, after the review of revision 1. The owner answered the
first question ("What else belongs in the public interface? `GraphData` and
`SimProcGraphData`, `Client`, `ExperimentStatus`, and the public exceptions:
A: add them. B: keep the list as proposed.") in their own words:

> Let's first think about the three types of users: the Model Developer creates a simulation model and defines data structure. The Experimentor installs the model, prepares the scenario data, and runs the simulation. The Operator installs the server and administers it.
>
> The Model Developer uses the disco module mostly via the Node interface. GraphData and SimProcGraphData are only accessed in a Node implementation via self.data. So none of the users has a need to import them.
>
> The Experimentor uses more of the disco interface. See the test_my_model.py file in the examples/my_model/tests folder. The Experimentor would use the Client. ExperimentStatus would only be used via that interface. So I think including it n the disco interface directly is not necessary.

Asked about the names that answer left open (the public exceptions,
`AppSettings`, `SpectralClusteringPartitioner`), the owner answered:

> Public Exceptions, AppSettings and SpectralClusteringPartitioner are all used by the experimentor.

**What exists today** (surveyed 2026-11-27, at `staging` `c6364f7`):
- **Layout.** `src/disco/` has 19 top-level modules (the largest
  `worker.py`, 1,255 lines) and nine subpackages: `cli`, `event_queue`,
  `experiments`, `graph`, `helpers`, `metastore`, `model`, `partitioner`,
  `transports`.
- **No public interface.** `disco/__init__.py` exports only `__version__`.
  Model authors import from wherever a name lives: the example model
  (`examples/my-model`) imports `Event` and `Node` from `disco.node`, and its
  test imports from `disco.config`, `disco.database`, `disco.experiments`,
  `disco.graph`, `disco.graph_factory`, `disco.model`, `disco.partitioner`
  and `disco.testrun`. Model plugins register in the `disco.models`
  entry-point group (`disco/model/loader.py:270`).
- **Operators** use the console scripts `disco` (`disco.__main__:main`) and
  `disco-server` (`disco.cli.server:main`) (`pyproject.toml:179-181`), plus
  `config.toml` and environment variables read by `disco.config`.
- **Import cycles** (carry-over 7):
  - `node` → `graph_data` (`node.py:13`, under `TYPE_CHECKING`) →
    `model` (`graph_data.py:29`) → `node` (`model/loader.py:17`;
    `loader.py:149` repeats the import locally: "local import avoids circular
    import at import-time").
  - `transports` → `runtime` (`transports/inprocess.py:11`,
    `transports/ipc_receiver.py:13`) → `router` (`runtime.py:39`) →
    `transports.base` (`router.py:10`): a cycle between packages, not
    between modules.
  - `transports.grpc_ingress` → `worker` (`grpc_ingress.py:30-31`, under
    `TYPE_CHECKING`) → `transports.grpc_ingress` (`worker.py:30`).
  - Only the control plane has a total order today: the `layers` contract
    `cli > server > orchestrator | worker > cluster > metastore`.
- **Layering contracts** (carry-over 6): `.importlinter` has 8 contracts,
  all kept. Six of them list their source or forbidden modules by hand
  (`only-config-reads-env`, `metastore-is-a-base`,
  `graph-is-self-contained`, `data-plane-below-the-entry-points`,
  `grpc-only-in-transports`, `kazoo-only-in-the-metastore`), so a new
  module is not covered until someone adds it.
- **The graph core reaches the database** (carry-over 3, decision 0001):
  `graph/core.py` imports nothing from SQLAlchemy itself, but it imports
  `GraphMask` (`core.py:14`), and `graph_mask.py` imports SQLAlchemy and
  `schema.vertex_masks` (`graph_mask.py:8-11`) for its persistence methods.
  The contract `graph-core-is-database-agnostic` therefore sets
  `allow_indirect_imports = True`; decision 0001 says removing that flag is
  the proof the restructuring has fixed it.
- **Fixed paths a move must keep working:** `transport.proto` and its
  generated stubs (`src/disco/transports/proto/`, regenerated with
  `-I src`); `src/disco/_version.py` (written by setuptools_scm, imported by
  `disco/__init__.py:2`); the C++ extension (`add_subdirectory(src/disco/event_queue)`,
  target `event_queue_core`, installed to `disco/event_queue` as `_core`);
  `wheel.packages = ["src/disco"]`, the console scripts, and the mypy and
  ruff settings in `pyproject.toml`. The workflows name no module paths.
- **Type checking:** `mypy --strict` reports 37 errors in 10 files today
  (`type-arg` 26, `no-untyped-call` 11); the strict switch is carry-over 2,
  a follow-up to this requirement.

## Rules

1. **disco has one documented public interface** (owner). Model authors
   import what they use from `disco` itself, and from a small set of named
   public subpackages. The interface is written down in the README, and in
   each public package's `__all__`; it is what "stable" will mean in the
   first stable release.
   - The proposed contents, for the owner's approval with this
     requirement *(architect)*: from `disco`: `Node`, `Event`, `TestRun`,
     `Experiment`, `load_model`, `graph_from_model`, the partitioners
     (`Partitioner`, `SimplePartitioner`, `SpectralClusteringPartitioner`),
     `SessionManager`, `get_settings` and `AppSettings`, and `__version__`;
     public subpackages: `disco.graph` (scenario storage and data access:
     today's `disco.graph.__all__`) and `disco.model` (the plugin API for
     model packages). The branch may add a name that a model author
     demonstrably needs (the example model, the README, the tests of the
     public interface); each addition is recorded, and none is removed
     without the owner.
   - A test fails if a name in a public `__all__` cannot be imported from
     where the interface says, or if the example model imports anything from
     outside the interface.
2. **Everything outside the public interface is internal** (owner). Its
   modules may be moved, renamed and split freely, now and later, and the
   layout says so: an internal module cannot be mistaken for a public one
   *(architect: by a leading underscore or an `_internal` package; the branch
   decides and records which)*.
3. **Old import paths stop working in the restructuring release** (owner).
   No compatibility shims or deprecation warnings. The README gets a table
   from every import path the example model, the README and `ENGINEERING.md`
   used before to its replacement.
4. **Operators see no change** (owner). The console scripts `disco` and
   `disco-server` keep their names, arguments and behaviour; every
   `config.toml` key and environment variable name `disco.config` reads
   today is read the same way after. A test pins the settings' keys and
   environment names against today's, so a rename fails it.
5. **No behaviour changes** *(architect)*. The restructuring moves code; it
   does not change what any function does. The whole suite passes with test
   changes limited to import paths, plus tests for the new rules; a test
   whose assertions change is a finding. Decision 0002 applies: nothing is
   re-implemented on the way.
6. **No import cycles** *(architect)*, closing carry-over 7. No `disco`
   module imports, directly or indirectly, a module that imports it back,
   counting imports under `TYPE_CHECKING`. The three cycles listed above are
   gone, without local imports that only hide a cycle (as `loader.py:149`
   does today).
7. **The whole package is layered, and the layering is checked** *(architect)*.
   Every top-level module and subpackage sits in one order, from the entry
   points down to the bases, checked by an import-linter `layers` contract
   (or contracts) in CI; the order is recorded in `ENGINEERING.md`.
8. **A new module is covered without editing the contracts** *(architect)*,
   closing carry-over 6. Every contract applies to a module added anywhere
   under `disco` without anyone listing it. A test, or the contracts' own
   design, fails when a module exists that no contract covers.
9. **The graph core is database-agnostic, enforced directly** *(architect)*,
   closing carry-over 3. Nothing `disco.graph.core` imports, directly or
   indirectly, imports SQLAlchemy: graph masks keep their vector logic in
   the core and their persistence beside the other database code. The
   contract `graph-core-is-database-agnostic` loses
   `allow_indirect_imports = True` and still passes; decision 0001 is
   marked superseded.
10. **Generated and native files keep working** *(architect)*. If a move
    reaches `transport.proto`, `_version.py` or the `event_queue` extension,
    the stubs are regenerated with the documented command (never edited),
    the CMake install destination follows the module, and the wheels build
    and pass on all three platforms (the release PR's `build.yml` run).
11. **No contract or check is loosened** *(architect)*. No lint rule,
    import contract or mypy setting is relaxed to make the move pass
    (CLAUDE.md: always escalate).

## Out of scope

- The two slow data paths, carry-overs 23 and 24: their own requirement
  after this one (owner: "A").
- Strict type checking, carry-over 2: a follow-up after this requirement
  (owner's topic: "mypy strict, item 2, follows").
- Any change to the wire format (`transport.proto` messages, envelopes),
  file formats, or what data leaves the process.
- Deprecation shims for old import paths (owner: "A").
- New features or API additions beyond gathering today's names into the
  public interface.

## How the work is split *(architect)*

- **01a, graph core without the database:** rule 9, and rules 5, 10 and 11
  for its files. Confined to `disco.graph`, so it can run in parallel with
  01b; it lands first or merges `origin/staging` before 01b moves anything
  under `graph`.
- **01b, the public interface and the internal layout:** rules 1–5, 10, 11.
  The moves that model authors see, the README's interface and old-to-new
  table, and the operator pin of rule 4. Stacked after nothing; its review
  checks the interface against the example model.
- **01c, cycles and a checked total layering:** rules 6, 7 and 8, and 5, 11.
  Cut after 01b lands, because it layers the modules in their new places.
  It rewrites `.importlinter`.

## What this changes in `CLAUDE.md`, `ENGINEERING.md` and the user guide

- `README.md`: the public interface (rule 1), a note that everything else
  is internal (rule 2), and the old-to-new import table (rule 3).
- `ENGINEERING.md`: each chapter's module names follow the moves; Ch 7
  states where graph-mask persistence lives (rule 9); a section records the
  package's layer order (rule 7) and the public interface.
- `CLAUDE.md`: "What this is" and the invariants name the new module paths
  (the graph-core invariant included).
- `docs/decisions/0001-*`: marked superseded once rule 9's contract holds.
- `.importlinter`: rewritten on 01c (rules 7, 8), its header comment updated.
- `docs/carry-overs.md`: items 1, 3, 6 and 7 deleted by the branches that
  close them.
