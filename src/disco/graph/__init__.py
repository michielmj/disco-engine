# src/disco/graph/__init__.py
"""
disco.graph
===========

Layered graph + scenario subsystem.

Public API (this subpackage):

- Graph                  : core in-memory graph structure (layers, vertices, labels, mask, vertex_weight).
- SuperGraph             : compressed graph produced by Graph.compress(); supervertices carry
                           summed vertex_weight from their constituent original vertices.
- GraphMask              : graph mask object.
- create_graph_schema    : create the graph schema/tables in a database.
- store_graph            : validate and persist a Graph (scenario + vertices + edges + labels)
                           to the DB.
- load_graph_for_scenario: load a Graph (vertices + edges + labels) from the DB for a scenario.

- get_node_types         : return node_type per vertex index from graph.vertices,
                           optionally filtered by a mask.
- get_vertex_data        : join model vertex tables (key-based) to graph.vertices
                           (index-based) and return a DataFrame.
- get_vertex_numeric_vector
                         : extract a numeric vertex column as a GraphBLAS Vector.
- get_outbound_edge_data : join model edge tables (key-based) to structural edges
                           and return a DataFrame (outbound semantics).
- get_inbound_edge_data  : same as above but inbound semantics.
- get_outbound_map       : DB-backed outbound adjacency Matrix for a given layer,
                           optionally masked.
- get_inbound_map        : DB-backed inbound adjacency Matrix for a given layer,
                           optionally masked.

Higher-level facades such as GraphData / SimProcGraphData live in the
top-level `disco.graph_data` module and are not part of this subpackage.
"""

from __future__ import annotations

from .core import Graph, SuperGraph
from .schema import create_graph_schema
from .db import (
    store_graph,
    load_graph_for_scenario,
)
from .extract import (
    get_vertex_data,
    get_vertex_numeric_vector,
    get_inbound_edge_data,
    get_inbound_map,
    get_outbound_edge_data,
    get_outbound_map,
)
from .graph_mask import GraphMask

__all__ = [
    "Graph",
    "SuperGraph",
    "GraphMask",
    "create_graph_schema",
    "store_graph",
    "load_graph_for_scenario",
    "get_vertex_data",
    "get_vertex_numeric_vector",
    "get_outbound_map",
    "get_outbound_edge_data",
    "get_inbound_map",
    "get_inbound_edge_data",
]
