# src/disco/testrun.py
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Generator, Tuple

from data_logger import DataLogger
from numpy.random import SeedSequence

from tools.mp_logging import getLogger

from .config import AppSettings
from .exceptions import DiscoError
from .experiments import Experiment, ExperimentStatus
from .graph import Graph
from .graph_data import GraphData
from .model import Model, load_model
from .partitioner import SimplePartitioner
from .partitioning import Partitioning
from .router import Router
from .runtime import NodeRuntime
from .transports.inprocess import InProcessTransport
from .database import SessionManager

logger = getLogger(__name__)


class TestRunError(DiscoError):
    """Base exception for TestRun-related failures."""


class TestRun:
    """
    TestRun executes a single, unpartitioned simulation run for debugging and unit tests.

    Key properties:
    - No Metastore / no Cluster registration
    - No threading
    - In-process transport only
    - No event/promise ingress queues
    - Exceptions are NOT caught (debugging aid)
    - Experiment status is updated directly on the Experiment object

    Lifecycle:
      ctor:
        - validates inputs (incl. partitioning=single partition)
        - builds Router and NodeRuntimes (but does not initialize them)
        - prepares deterministic per-node SeedSequences
      initialize():
        - calls NodeRuntime.initialize() for each node in the order defined by partitioning.node_specs
        - sets ExperimentStatus.INITIALIZED
      run(duration):
        - ensures initialized
        - sets ExperimentStatus.ACTIVE
        - steps per-node runners until all StopIteration for the given duration
        - does not tear down; can be called again
    """

    def __init__(
        self,
        *,
        experiment: Experiment,
        graph: Graph,
        settings: AppSettings,
        model: Model | None = None,
        dlogger: DataLogger | None = None,
        partitioning: Partitioning | None = None,
        master_seed: int | SeedSequence | None = None,
    ) -> None:
        self._experiment = experiment
        self._graph = graph
        self._settings = settings

        self._session_manager = SessionManager.from_settings(settings.database)

        if model is None:
            # Model is a fixed package; load once. Errors should raise (debugging).
            ms = self._settings.model
            self._model: Model = load_model(
                plugin=ms.plugin,
                package=ms.package,
                path=ms.path,
                db=self._session_manager,
                dev_import_root=ms.dev_import_root,
                model_yml=ms.model_yml,
            )
        else:
            self._model = model

        if experiment.replications:
            repid = next(iter(experiment.replications.keys()))
        else:
            repid = str(uuid.uuid4())

        # build dlogger if needed
        if dlogger is None:
            dl = settings.data_logger
            dl_path = Path(dl.path) / experiment.expid / repid
            self._dlogger = DataLogger(
                segments_dir=dl_path,
                ring_bytes=dl.ring_bytes,
                rotate_bytes=dl.rotate_bytes,
                zstd_level=dl.zstd_level,
            )
        else:
            self._dlogger = dlogger

        # build partitioning if needed
        if partitioning is None:
            partitioner = SimplePartitioner(graph=graph, model=self._model)
            self._partitioning = partitioner.partition(1)
        else:
            self._partitioning = partitioning

        if self._partitioning.num_partitions != 1:
            raise TestRunError('A partitioning with more than one partition is not allowed for a test run.')

        if not self._partitioning.node_specs:
            raise TestRunError("TestRun requires at least one node.")

        # Seed generation:
        # - master_seed may be int or SeedSequence
        # - spawn exactly len(node_names) SeedSequences
        if master_seed is None:
            master_seed = 0
        ss = SeedSequence(master_seed) if isinstance(master_seed, int) else master_seed
        self._master_seed = ss
        node_ss = ss.spawn(len(self._partitioning.node_specs))

        # Router + in-process transport:
        self.nodes: dict[str, NodeRuntime] = {}  # InProcessTransport keeps a reference
        self._router = Router(
            transports=[InProcessTransport(nodes=self.nodes)],
        )

        # Build NodeRuntimes now (as requested), but do not initialize them yet.
        self._initialized = False

        for i, ns in enumerate(self._partitioning.node_specs):

            node_mask = self._partitioning.assignment_vector(ns.node_name)
            graph_view = graph.get_view(mask=node_mask)

            graph_data = GraphData.for_node(
                session_manager=self._session_manager,
                graph=graph_view,
                model=self._model,
                partitioning=self._partitioning,
                node_name=ns.node_name
            )

            rt = NodeRuntime(
                repid=repid,
                spec=ns,
                model=self._model,
                partitioning=self._partitioning,
                router=self._router,
                dlogger=self._dlogger,
                seed_sequence=node_ss[i],  # SeedSequence per node (deterministic assignment)
                graph=graph_view,
                data=graph_data,
            )

            # Insert into dict for InProcessTransport delivery.
            self.nodes[ns.node_name] = rt

        self._experiment.status = ExperimentStatus.LOADED

        logger.info(
            "TestRun created: repid=%s scenario=%s with nodes:\n%s",
            repid,
            graph.scenario_id,
            "\n".join([ns.node_name for ns in self._partitioning.node_specs])
        )

    @property
    def experiment(self) -> Experiment:
        return self._experiment

    @property
    def graph(self) -> Graph:
        return self._graph

    @property
    def model(self) -> Model:
        return self._model

    @property
    def router(self) -> Router:
        return self._router

    @property
    def node_names(self) -> list[str]:
        return [ns.node_name for ns in self._partitioning.node_specs]

    def initialize(self) -> None:
        """
        Initialize all NodeRuntimes in deterministic order (by partitioning.node_specs).

        Sets experiment status: LOADED -> INITIALIZED

        Does not catch exceptions (by design for debugging).
        """
        if self._initialized:
            return

        params = self._experiment.params or {}
        for ns in self._partitioning.node_specs:
            self.nodes[ns.node_name].initialize(**params)

        self._experiment.status = ExperimentStatus.INITIALIZED
        self._initialized = True

    def run(self, duration: float) -> None:
        """
        Run the simulation up to 'duration' (simulation time horizon).

        - Ensures nodes are initialized (initialize() will be called if needed).
        - Sets experiment status to ACTIVE (and leaves it ACTIVE).
        - Steps per-node runners until all StopIteration for the given duration.
        - Does not tear down; can be called multiple times.

        Exceptions are not caught; any error is raised to aid debugging.
        """
        if not self._initialized:
            self.initialize()

        self._experiment.status = ExperimentStatus.ACTIVE

        # Create one runner per node for this duration.
        runners: Tuple[Generator[object, None, None], ...] = tuple(
            self.nodes[ns.node_name].runner(duration=duration) for ns in self._partitioning.node_specs
        )
        active = list(range(len(runners)))

        # Round-robin stepping until all are done for this duration.
        finished_positions: list[int] = []
        while active:
            for pos in range(len(active)):
                idx = active[pos]
                try:
                    next(runners[idx])
                except StopIteration:
                    finished_positions.append(pos)

            # Remove in reverse order to keep indices valid.
            for pos in reversed(finished_positions):
                active.pop(pos)

            finished_positions.clear()

        self._dlogger.close()
