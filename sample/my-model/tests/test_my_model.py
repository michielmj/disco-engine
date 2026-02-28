import os

from data_logger import DataLogger

from disco.graph import load_graph_for_scenario
from disco.model import load_model
from disco.config import get_settings
from disco.partitioner import SimplePartitioner
from disco.database import SessionManager
from disco.experiments import Experiment
from disco.testrun import TestRun


def test_run(tmp_path):
    print(os.curdir)
    settings = get_settings()
    scenario_id = 'demo'
    smgr = SessionManager.from_settings(settings.database)
    with smgr.session() as session:
        graph = load_graph_for_scenario(session, scenario_id)

    model = load_model(plugin='my-model')

    partitioner = SimplePartitioner(graph, model)
    partitioning = partitioner.partition(1)

    experiment = Experiment(100., 'demo')
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

    testrun.run(10)



