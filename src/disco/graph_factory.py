import numpy as np
import pandas as pd

from sqlalchemy import select, literal
from disco.model import Model, db_validate
from disco.graph import Graph
from disco.database import DbHandle
from disco.partitioner import NODE_TYPE


def graph_from_model(db: DbHandle, scenario_id: str, model: Model) -> Graph:

    spec = model.spec
    orm = model.orm

    db_validate(orm, db)

    with db.session() as session:

        # build index map and labels
        keys = list()
        labels = {NODE_TYPE: dict()}

        for nt, ns in spec.node_types.items():
            table_name = ns.node_data_table
            table = model.orm.node_tables[table_name]

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
            vertices['index'] = np.arange(offset, offset + vertices.shape[0])
            keys += vertices['key'].tolist()

            labels[NODE_TYPE].update({nt: vertices['index'].array})
            for lt in ns.distinct_nodes:
                if lt not in labels:
                    labels[lt] = dict()

                for lbl, group in vertices.groupby(lt)['index']:
                    if lbl in labels[lt]:
                        labels[lt][lbl] = np.union1d(labels[lt][lbl], group.array)
                    else:
                        labels[lt][lbl] = group.array

        keymap = {k: i for i, k in enumerate(keys)}
        num_vertices = len(keys)

        # build layers
        layers = list()
        for simproc in spec.simprocs:
            if simproc in orm.edge_tables_by_simproc:
                table = orm.edge_tables_by_simproc[simproc]
            else:
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
                edges['source_key'].map(keymap),
                edges['target_key'].map(keymap),
                np.zeros(edges.shape[0], dtype=float)
            ))

        graph = Graph.from_edges(
            edge_layers={i: l for i, l in enumerate(layers)},
            num_vertices=num_vertices
        )
        for lt, lbl in labels:
            graph.add_labels(lt, lbl)

    return graph
