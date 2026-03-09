import os

from data_logger import DataLogger

from disco.graph import store_graph
from disco.model import load_model
from disco.config import get_settings
from disco.partitioner import SimplePartitioner
from disco.database import SessionManager
from disco.experiments import Experiment
from disco.testrun import TestRun
from disco.graph_factory import graph_from_model


def test_run(tmp_path):
    print(os.curdir)
    settings = get_settings()
    scenario_id = 'demo'
    smgr = SessionManager.from_settings(settings.database)
    model = load_model(plugin="my-model")
    # with smgr.session() as session:
    #     graph = load_graph_for_scenario(session, scenario_id)
    graph = graph_from_model(smgr, "demo", model)
    with smgr.session() as session:
        store_graph(session, graph, replace=True)

    partitioner = SimplePartitioner(graph, model)
    partitioning = partitioner.partition(1)

    experiment = Experiment(
        duration=100,
        scenario_id="demo",
        allowed_partitionings=[partitioning.partitioning_id]
    )
    dlogger = DataLogger(tmp_path / 'dlogger')

    testrun = TestRun(
        experiment=experiment,
        graph=graph,
        settings=settings,
        model=model,
        dlogger=dlogger,
        partitioning=partitioning
    )
    testrun.initialize()

    testrun.run(100)



