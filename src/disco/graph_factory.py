from typing import List, Dict
import numpy as np
import pandas as pd

from sqlalchemy import select, literal
from disco.model import Model, db_validate
from disco.graph import Graph
from disco.database import DbHandle, SessionManager, normalize_db_handle
from disco.partitioner import NODE_TYPE


def graph_from_model(db: DbHandle, scenario_id: str, model: Model) -> Graph:

    smgr = SessionManager(engine=normalize_db_handle(db))

    spec = model.spec
    orm = model.orm

    db_validate(orm, db)

    with smgr.session() as session:

        # build index map and labels
        keys: List[str] = []
        labels: Dict[str, Dict[str, np.ndarray]] = {NODE_TYPE: dict()}

        for nt, ns in spec.node_types.items():
            table = orm.node_tables[nt]  # keyed by node-type name, not table name

            column_names = ['key'] + list(ns.distinct_nodes)

            cursor = session.execute(
                select(
                    *[table.c[name] for name in column_names]
                ).where(
                    table.c['scenario_id'] == literal(scenario_id)
                )
            )
            vertices = pd.DataFrame(cursor.fetchall(), columns=column_names)

            offset = len(keys)
            vertices['index'] = np.arange(offset, offset + len(vertices))
            keys += vertices['key'].tolist()

            labels[NODE_TYPE].update({nt: vertices['index'].to_numpy()})
            for lt in ns.distinct_nodes:
                if lt not in labels:
                    labels[lt] = dict()

                for lbl, group in vertices.groupby(lt)['index']:
                    grp_arr = group.to_numpy()
                    if lbl in labels[lt]:
                        labels[lt][lbl] = np.union1d(labels[lt][lbl], grp_arr)
                    else:
                        labels[lt][lbl] = grp_arr

        keymap = {k: i for i, k in enumerate(keys)}
        num_vertices = len(keys)

        # build layers
        layers = list()
        for simproc in spec.simprocs:
            if simproc in orm.edge_tables_by_simproc:
                table = orm.edge_tables_by_simproc[simproc]
            else:
                if orm.default_edge_table is None:
                    raise ValueError(
                        f"Simproc '{simproc}' has no dedicated edge table and "
                        "no default_edge_data_table is configured in the model."
                    )
                table = orm.default_edge_table

            cursor = session.execute(
                select(
                    table.c['source_key'],
                    table.c['target_key']
                ).where(
                    table.c['scenario_id'] == literal(scenario_id)
                )
            )
            edges = pd.DataFrame(cursor.fetchall(), columns=['source_key', 'target_key'])

            layers.append((
                edges['source_key'].map(keymap).to_numpy(dtype=np.int64),
                edges['target_key'].map(keymap).to_numpy(dtype=np.int64),
                np.ones(len(edges), dtype=float),
            ))

        graph = Graph.from_edges(
            edge_layers={i: layer for i, layer in enumerate(layers)},
            num_vertices=num_vertices,
            scenario_id=scenario_id,
        )
        for lt, lbl in labels.items():
            graph.add_labels(lt, lbl)

    return graph
