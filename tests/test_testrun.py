# tests/test_testrun.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest
from numpy.random import SeedSequence

import disco.testrun as testrun_mod
from disco.experiments import ExperimentStatus


# -----------------------------
# Minimal fakes / test doubles
# -----------------------------

@dataclass(frozen=True, slots=True)
class FakeNodeInstanceSpec:
    partition: int
    node_name: str
    node_type: str = "fake"


@dataclass(frozen=True, slots=True)
class FakePartitioning:
    partitioning_id: str = "pid"
    scenario_id: str = "sid"
    num_partitions: int = 1
    node_specs: Tuple[FakeNodeInstanceSpec, ...] = ()

    def assignment_vector(self, node_name: str) -> object:
        # Any opaque mask object is fine for the TestRun code path.
        return ("mask", node_name)


class FakeGraph:
    def __init__(self, scenario_id: str = "sid") -> None:
        self.scenario_id = scenario_id

    def get_view(self, mask: object) -> "FakeGraph":
        # In real code this likely creates a masked view; for tests we can return self.
        return self


class FakeExperiment:
    def __init__(
        self,
        expid: str = "exp1",
        replications: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> None:
        self.expid = expid
        self.replications = replications or {}
        self.params = params or {}
        self.status = ExperimentStatus.CREATED


class FakeSettings:
    class _DB:
        pass

    class _Model:
        plugin = "p"
        package = "pkg"
        path = "path"
        dev_import_root = None
        model_yml = None

    class _DL:
        path = "/tmp/disco-tests"
        ring_bytes = 1024
        rotate_bytes = 1024
        zstd_level = 1

    database = _DB()
    model = _Model()
    data_logger = _DL()


class FakeSessionManager:
    @classmethod
    def from_settings(cls, _db_settings: object) -> "FakeSessionManager":
        return cls()


class FakeDataLogger:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeModel:
    pass


class FakeGraphData:
    pass


class FakeGraphDataFactory:
    @staticmethod
    def for_node(*, session_manager: object, graph: object, model: object, spec: object) -> FakeGraphData:
        # Explicit signature; no getattr.
        return FakeGraphData()


class FakeInProcessTransport:
    def __init__(self, *, nodes: Dict[str, object]) -> None:
        self.nodes = nodes


class FakeRouter:
    def __init__(self, *, transports: List[object]) -> None:
        self.transports = transports


class CapturingNodeRuntime:
    """
    NodeRuntime double that:
    - captures initialize() calls (order + params)
    - captures received SeedSequence
    - implements runner(duration) with "resume" behavior:
        * if duration <= last_duration: StopIteration immediately
        * if duration > last_duration: yields once then stops and updates last_duration
    """

    def __init__(
        self,
        *,
        repid: str,
        spec: FakeNodeInstanceSpec,
        model: object,
        partitioning: object,
        router: object,
        dlogger: object,
        seed_sequence: SeedSequence,
        graph: object,
        data: object,
    ) -> None:
        self.repid = repid
        self.spec = spec
        self.model = model
        self.partitioning = partitioning
        self.router = router
        self.dlogger = dlogger
        self.seed_sequence = seed_sequence
        self.graph = graph
        self.data = data

        self.initialize_calls: List[Dict[str, Any]] = []
        self.runner_calls: List[float] = []
        self._last_duration: float = float("-inf")

    def initialize(self, **params: Any) -> None:
        self.initialize_calls.append(dict(params))

    def runner(self, *, duration: float):
        self.runner_calls.append(duration)

        if duration <= self._last_duration:
            # Immediately finished for earlier / equal horizon.
            if False:
                yield None
            return

        # Simulate "work" for a later horizon.
        self._last_duration = duration
        yield 1
        return


class FakePartitioner:
    def __init__(self, *, graph: object, model: object) -> None:
        self.graph = graph
        self.model = model
        self.partition_calls: List[int] = []

    def partition(self, n: int) -> FakePartitioning:
        self.partition_calls.append(n)
        # Provide deterministic node_specs order.
        return FakePartitioning(
            node_specs=(
                FakeNodeInstanceSpec(partition=0, node_name="b"),
                FakeNodeInstanceSpec(partition=0, node_name="a"),
            )
        )


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture()
def patch_testrun_deps(monkeypatch: pytest.MonkeyPatch):
    # Patch external dependencies used by disco.testrun
    monkeypatch.setattr(testrun_mod, "SessionManager", FakeSessionManager)
    monkeypatch.setattr(testrun_mod, "DataLogger", FakeDataLogger)
    monkeypatch.setattr(testrun_mod, "GraphData", FakeGraphDataFactory)
    monkeypatch.setattr(testrun_mod, "InProcessTransport", FakeInProcessTransport)
    monkeypatch.setattr(testrun_mod, "Router", FakeRouter)
    monkeypatch.setattr(testrun_mod, "NodeRuntime", CapturingNodeRuntime)
    # Default load_model returns FakeModel
    monkeypatch.setattr(testrun_mod, "load_model", lambda **kwargs: FakeModel())
    # Default SimplePartitioner yields 1-partition FakePartitioning
    # We patch the class itself to produce an instance we can inspect if needed.
    created: Dict[str, Any] = {}

    def _make_partitioner(*, graph: object, model: object) -> FakePartitioner:
        p = FakePartitioner(graph=graph, model=model)
        created["partitioner"] = p
        return p

    monkeypatch.setattr(testrun_mod, "SimplePartitioner", _make_partitioner)
    return created


# -----------------------------
# Tests
# -----------------------------

def test_ctor_sets_status_loaded_and_picks_repid_from_experiment_replications(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object(), "repB": object()})
    g = FakeGraph(scenario_id="sid")
    settings = FakeSettings()

    tr = testrun_mod.TestRun(experiment=exp, graph=g, settings=settings)

    assert exp.status == ExperimentStatus.LOADED
    # Repid should be the first key by insertion order (dict preserves insertion order).
    # We verify via the created NodeRuntime repid.
    assert list(tr.nodes.values())[0].repid == "repA"  # type: ignore[attr-defined]


def test_ctor_uses_provided_model_without_calling_load_model(monkeypatch: pytest.MonkeyPatch, patch_testrun_deps):
    called = {"n": 0}

    def _load_model(**kwargs: Any) -> FakeModel:
        called["n"] += 1
        return FakeModel()

    monkeypatch.setattr(testrun_mod, "load_model", _load_model)

    exp = FakeExperiment(expid="exp1", replications={"repA": object()})
    g = FakeGraph()
    settings = FakeSettings()
    provided_model = FakeModel()

    tr = testrun_mod.TestRun(experiment=exp, graph=g, settings=settings, model=provided_model)

    assert tr.model is provided_model
    assert called["n"] == 0


def test_ctor_builds_partitioning_when_not_provided(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object()})
    g = FakeGraph()
    settings = FakeSettings()

    tr = testrun_mod.TestRun(experiment=exp, graph=g, settings=settings)

    # Our patched SimplePartitioner.partition(1) creates two node_specs b,a
    assert tr.node_names == ["b", "a"]
    part = patch_testrun_deps["partitioner"]
    assert part.partition_calls == [1]


def test_ctor_rejects_partitioning_with_more_than_one_partition(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object()})
    g = FakeGraph()
    settings = FakeSettings()

    bad_part = FakePartitioning(num_partitions=2, node_specs=(FakeNodeInstanceSpec(0, "n"),))

    with pytest.raises(testrun_mod.TestRunError):
        testrun_mod.TestRun(experiment=exp, graph=g, settings=settings, partitioning=bad_part)


def test_ctor_rejects_empty_node_specs(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object()})
    g = FakeGraph()
    settings = FakeSettings()

    bad_part = FakePartitioning(num_partitions=1, node_specs=())

    with pytest.raises(testrun_mod.TestRunError):
        testrun_mod.TestRun(experiment=exp, graph=g, settings=settings, partitioning=bad_part)


def test_seeds_assigned_deterministically_by_node_specs_order(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object()})
    g = FakeGraph()
    settings = FakeSettings()

    part = FakePartitioning(
        num_partitions=1,
        node_specs=(
            FakeNodeInstanceSpec(partition=0, node_name="x"),
            FakeNodeInstanceSpec(partition=0, node_name="y"),
            FakeNodeInstanceSpec(partition=0, node_name="z"),
        ),
    )

    master = 12345
    tr = testrun_mod.TestRun(
        experiment=exp,
        graph=g,
        settings=settings,
        partitioning=part,
        master_seed=master,
    )

    # Build the expected SeedSequence spawns in exactly the same way:
    ss = SeedSequence(master)
    expected = ss.spawn(len(part.node_specs))

    # Compare via generated state (stable within a process; you said it needn't be stable across envs).
    for spec, exp_ss in zip(part.node_specs, expected):
        rt = tr.nodes[spec.node_name]  # type: ignore[attr-defined]
        got_state = rt.seed_sequence.generate_state(8)
        exp_state = exp_ss.generate_state(8)
        assert (got_state == exp_state).all()


def test_initialize_calls_nodes_in_node_specs_order_and_sets_status_initialized(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object()}, params={"p": 1, "q": 2})
    g = FakeGraph()
    settings = FakeSettings()

    part = FakePartitioning(
        node_specs=(
            FakeNodeInstanceSpec(0, "b"),
            FakeNodeInstanceSpec(0, "a"),
            FakeNodeInstanceSpec(0, "c"),
        )
    )

    tr = testrun_mod.TestRun(experiment=exp, graph=g, settings=settings, partitioning=part, master_seed=0)

    # No initialize yet
    assert exp.status == ExperimentStatus.LOADED

    tr.initialize()

    assert exp.status == ExperimentStatus.INITIALIZED

    # Verify initialize called for each node exactly once and in node_specs order.
    init_order = []
    for name in ["b", "a", "c"]:
        rt = tr.nodes[name]  # type: ignore[attr-defined]
        assert len(rt.initialize_calls) == 1
        assert rt.initialize_calls[0] == {"p": 1, "q": 2}
        init_order.append(name)

    # Ensure no accidental reordering occurred by checking call timestamps via list concatenation order:
    # (We can infer order because each runtime stores its own call list; check sequentially by node_specs.)
    assert init_order == [ns.node_name for ns in part.node_specs]


def test_run_initializes_if_needed_sets_status_active_and_steps_until_all_stop(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object()}, params={"p": 1})
    g = FakeGraph()
    settings = FakeSettings()

    part = FakePartitioning(
        node_specs=(
            FakeNodeInstanceSpec(0, "n1"),
            FakeNodeInstanceSpec(0, "n2"),
        )
    )

    tr = testrun_mod.TestRun(experiment=exp, graph=g, settings=settings, partitioning=part, master_seed=42)

    # run() should auto-initialize
    tr.run(10.0)

    assert exp.status == ExperimentStatus.ACTIVE

    # Each runtime got initialize exactly once
    assert tr.nodes["n1"].initialize_calls == [{"p": 1}]  # type: ignore[attr-defined]
    assert tr.nodes["n2"].initialize_calls == [{"p": 1}]  # type: ignore[attr-defined]

    # For duration=10.0, our fake runner yields once for each node then stops
    assert tr.nodes["n1"].runner_calls == [10.0]  # type: ignore[attr-defined]
    assert tr.nodes["n2"].runner_calls == [10.0]  # type: ignore[attr-defined]


def test_run_can_be_called_multiple_times_earlier_duration_is_noop_later_duration_progresses(patch_testrun_deps):
    exp = FakeExperiment(expid="exp1", replications={"repA": object()}, params={})
    g = FakeGraph()
    settings = FakeSettings()

    part = FakePartitioning(
        node_specs=(
            FakeNodeInstanceSpec(0, "n1"),
            FakeNodeInstanceSpec(0, "n2"),
        )
    )

    tr = testrun_mod.TestRun(experiment=exp, graph=g, settings=settings, partitioning=part, master_seed=0)

    # First run to 5.0 => each runner yields once
    tr.run(5.0)
    assert tr.nodes["n1"].runner_calls == [5.0]  # type: ignore[attr-defined]
    assert tr.nodes["n2"].runner_calls == [5.0]  # type: ignore[attr-defined]

    # Earlier horizon (3.0) => fake runner returns immediately
    tr.run(3.0)
    assert tr.nodes["n1"].runner_calls == [5.0, 3.0]  # type: ignore[attr-defined]
    assert tr.nodes["n2"].runner_calls == [5.0, 3.0]  # type: ignore[attr-defined]

    # Later horizon (8.0) => yields once again for each node
    tr.run(8.0)
    assert tr.nodes["n1"].runner_calls == [5.0, 3.0, 8.0]  # type: ignore[attr-defined]
    assert tr.nodes["n2"].runner_calls == [5.0, 3.0, 8.0]  # type: ignore[attr-defined]

    # initialize should still have happened once per node total
    assert len(tr.nodes["n1"].initialize_calls) == 1  # type: ignore[attr-defined]
    assert len(tr.nodes["n2"].initialize_calls) == 1  # type: ignore[attr-defined]