# disco's users: purpose, roles and use

The reference for who uses disco and how, for interface and requirement
work. The owner's words are quoted verbatim and dated; the rest is the
architect's reading of them, checked against the code and the example model
(`examples/my-model`). Correct it with the owner, never by inference.

## What disco is for

disco is a discrete-event simulation engine for **large models whose state
is expressed as sparse vectors**, run on one machine or across many. Its
main domains are supply chains and manufacturing.

> Disco is primarily meant for large models with states expressed as sparse vectors. The vectorised node is the way to make Python simulation performant. This is what sets it apart from SimPy that in my experience is not scalable and performant enough for large simulations (no abstractions, no native vector operation support). To process vectors efficiently, graphblas is the recommended toolbox. You may challenge this of course.
> (owner, 2026-11-27)

The concepts, as the engine uses them:

- **Scenario:** a graph stored in a database. Its vertices are the
  simulated entities' smallest units; each layer of edges is one kind of
  flow (orders upstream, goods downstream, ...).
- **Simproc:** one timeline per graph layer, in the order `model.yml`
  lists them.
- **Node:** a prototype of a processing entity, written by the Model
  Developer.

  > It is not precisely a group of vertices but a logical processing entity (a class). They are the basic building blocks of a simulation model. They describe prototypes. For example, a prototype customer, a prototype production line, a prototype stocking point. A Node instance can represent both multiple real entities together, or a part of a real entity. Vertices are dynamically assigned to nodes (node instances) in a way that balances processing and communication effort in the simulation run. This is the task of the partitioner.
  > (owner, 2026-11-27)

  So a node class's logic must hold for whatever set of vertices its
  instance is given; it works on them as vectors.
- **Partitioning:** the partitioner's allocation of vertices to node
  instances, and of node instances to workers.
- **Synchronisation:** conservative. Node instances send *promises*
  ("nothing before epoch t") to their successors, so each can advance
  safely; lead times give the lookahead that makes this efficient.
- **Reproducibility:**

  > No, because the simulation uses random number streams, it cannot guarantee the same result irrespective of the partitioning. But with the same data, model, parameters, partitioning, and seed, the result should be the same.
  > (owner, 2026-11-27, asked whether results should be the same however a run is partitioned)

  The partitioning is therefore part of what identifies a result. Across
  partitionings, results agree statistically, not exactly.

## The three roles

> Let's first think about the three types of users: the Model Developer creates a simulation model and defines data structure. The Experimentor installs the model, prepares the scenario data, and runs the simulation. The Operator installs the server and administers it.
> (owner, 2026-11-27)

### Model Developer

Builds a model as its own Python package (the example: `examples/my-model`):

- a manifest, `model.yml`: the simprocs in order, the node types (each a
  `Node` class, its node-data table, the vertex labels that keep vertices
  apart or together, its self-relations between simprocs), and the edge-data
  tables;
- the scenario data structure, as SQLAlchemy tables (`orm.py`);
- the node classes: `__init__(params)` reads the instance's data through
  `self.data`; `on_events(simproc, events)` reacts, using `send_event`,
  `wakeup` and `advance_promise`, `self.rng`, and `self.dlogger` for
  results;
- what results to collect, through disco-data-logger.

> The Model Developer uses the disco module mostly via the Node interface. GraphData and SimProcGraphData are only accessed in a Node implementation via self.data. So none of the users has a need to import them.
> (owner, 2026-11-27)

> Loading data in a database is outside the scope of disco-engine. The Model Developer needs to ensure the data has structure that is readable by disco and needs to specify this structure in the model manifest and orm. See the example project.
> (owner, 2026-11-27)

> Disco supports the collection of results via disco-data-logger. What is collected is up to the Model Developer. Refer to the documentation of disco-data-logger on github.
> (owner, 2026-11-27)

disco-data-logger (0.2.0, installed with disco) records labelled streams of
sparse vectors, periodic snapshots or accumulators, to compressed segment
files, and exports them to Arrow and Parquet for analysis after the run.

**Manufacturing, an open modelling problem:**

> I have used disco for manufacturing as well. The main challenge there is shared resources. For example, an overhead crane used in multiple processing steps. This tightly links nodes together, forcing them to stay synchronized. It requires a separate process layer for supporting reservation, allocation, and release of these resources. I have not yet thought of a good way to model this for disco.
> (owner, 2026-11-27)

> The problem is not that the crane cannot be a node of its own. In the layered DAG, it cannot be reserved, allocated, and released in the same process. Try to understand this mathematically.
> (owner, 2026-11-27, correcting the architect, who had suggested a crane node on its own simproc)

The owner's solution, as used so far:

> My solution to the shared resource problem has been: two flows (layers): a production flow, and an allocation flow, opposite directions. Node B is downstream of Node A in the production flow. SharedResource node is at the end. Node A sends a request to SharedResource on production flow. Node B sends a request to SharedResource on production flow. SharedResource receives requests and calculates its earliest availability on the production flow and sends a reservation confirmation on the allocation flow to Node A and to Node B. SharedResource sends allocation confirmation to Node A on allocation flow. After a period, Node A sends a release confirmation to SharedResource on production flow. SharedResource then sends allocation confirmation to Node B on allocation flow. However, the allocation flow cannot send events to its own production flow. It can store the allocations but the production flow will have to wake-up at regular intervals to see that it got allocated the resource. The wake-up interval will determine how frequently Node A, B and SharedResource will need to synchronize their time on the production flow. If Node A knows that it will use the resource for at least an amount of time, it can make an advance promise to the next node.
> (owner, 2026-11-27)

And, correcting the architect's restatement:

> No. A's production start is postponed each time with delta until the allocation is seen.
> (owner, 2026-11-27)

So the waiting is the model's logic: at each production wake-up A checks the
allocation it stored from the allocation flow, and if it is not there yet,
sets its next wake-up Δ later. The engine makes this safe: a node runtime
handles its simprocs strictly in epoch order (earliest next epoch first, the
higher simproc at a tie, and it waits while any simproc's next epoch is
unknown: `runtime.py` `runner`), so the stored allocations are complete up
to the epoch the production handler runs at.

### Experimentor

Installs the model, prepares the scenario data, and runs the simulation,
from Python scripts and notebooks. The steps, as
`examples/my-model/tests/test_my_model.py` takes them:

1. settings (`get_settings`) and a database session (`SessionManager`);
2. the model (`load_model`);
3. the scenario's data in the model's tables (loaded by the Experimentor's
   own means: outside disco), then the graph built from it
   (`graph_from_model`) and stored (`store_graph`);
4. a partitioning (`SimplePartitioner` or `SpectralClusteringPartitioner`);
5. an `Experiment`: duration, scenario, allowed partitionings, params,
   replications and seeds;
6. a run: locally with `TestRun`, or on a server through the `Client`
   (submit, monitor, pause, cancel);
7. the results, from disco-data-logger's output.

> The Experimentor uses more of the disco interface. See the test_my_model.py file in the examples/my_model/tests folder. The Experimentor would use the Client. ExperimentStatus would only be used via that interface. So I think including it n the disco interface directly is not necessary.
> (owner, 2026-11-27)

> Public Exceptions, AppSettings and SpectralClusteringPartitioner are all used by the experimentor.
> (owner, 2026-11-27)

> I imagine that Experimentors use both scripts and notebooks. TestRun is for local runs by Experimentors and Model Developers. Scaling up to larger scenarios, doing multiple experiments (param sweeps) and replications would then be done on the server.
> (owner, 2026-11-27)

### Operator

Installs the server and administers it: disco and the model packages on
each machine, ZooKeeper and the database, the configuration (`config.toml`,
environment variables, secrets), and `disco-server` (workers and the
orchestrator). The operator imports nothing from disco. For the
restructuring the owner decided that the operator sees no change: the
commands, configuration keys and environment names stay as they are
(requirement 01, round 1, answer "A").

Deployments vary without a typical size:

> That is impossible to answer in advance. It can be anything from one worker and one node class to tens or hundreds
> (owner, 2026-11-27)
