# A Distributed Simulation Core Engine
## 1. Overview

This document describes the architecture and design of **disco**: a **Distributed Simulation Core** for running large, event-driven simulations across multiple processes and machines.

At a high level, disco provides:

- A **runtime hierarchy** of:
  - **Application** → one OS process acting as the coordinator and host for multiple Workers.
  - **Workers** → long-lived simulation processes running on one or more machines.
  - **NodeRuntimes** → per-node managers that own simulation logic and event queues.
  - **SimProcs** → distinct simulation timelines.
  - **Nodes** state machines and simulation logic.
- A clear separation between:
  - **Control plane**: configuration, metadata, desired state, orchestration.
  - **Data plane**: events and promises flowing between nodes, Workers, and machines.
- A **metadata and coordination layer** built on ZooKeeper (wrapped by `Metastore` and `Cluster`), used for:
  - Worker registration and state
  - Experiment / replication / partition assignments
  - Routing metadata (which node lives where)
- A **routing and transport layer** that delivers events and promises:
  - In-process (same Worker)
  - Via IPC queues + shared memory (same machine, different process)
  - Via gRPC (different machines)
- A **graph/scenario subsystem** that stores layered DAGs in a relational DB, with in-memory acceleration using `python-graphblas` and persisted masks for efficient data extraction.

The **Graph** provides the **basic structure of a simulation scenario**: it describes which entities exist, how they are connected, and how interactions are layered. This graph structure **governs how nodes are created and how they interact**, and simprocs are closely linked to the graph layers. The exact mapping from graph to nodes and simprocs is specified in later chapters.

All Python source files live under:

- `src/disco/...`

The engineering spec is organized around these responsibilities:

- **Chapter 1–2** – High-level overview and core terminology.
- **Chapter 3–4** – Metastore (ZooKeeper) and Cluster (metadata, address book, control plane).
- **Chapter 5** – Worker architecture and lifecycle.
- **Chapter 6** – Routing and transport (InProcess, IPC, gRPC).
- **Chapter 7** – Layered graph and scenarios, plus higher-level modules as they are introduced.
- **Chapter 8** - Model definition
- **Chapter 9+** - Core simulation runtime blocks: EventQueue, SimProc, NodeRuntime
This spec focuses on architectural contracts and invariants rather than specific simulation models. Different simulation engines (domains, models) should be able to sit on top of the same core without changing the infrastructure components.

---

## 2. Concepts & Terminology

This section defines the core concepts used throughout the spec and codebase. Later chapters reference these terms without redefining them.

### 2.1 Runtime Hierarchy

- **Application**  
  A single OS process that bootstraps the system (e.g. a CLI or service entrypoint). An application:
  - Creates and manages one or more **Workers** (often via `multiprocessing`).
  - Instantiates a single **Metastore** and **Cluster** client in its own process.
  - Optionally runs a small simulation locally (e.g. for small experiments) without gRPC.

- **Worker**  
  A long-lived OS process that:

  - Hosts a set of **NodeRuntimes** for one or more nodes.
  - Runs a **single runner loop** on one thread for determinism.
  - Maintains a **WorkerState** (`CREATED`, `AVAILABLE`, `INITIALIZING`, `READY`, `ACTIVE`, `PAUSED`, `TERMINATED`, `BROKEN`).
  - Installs routing and transports for its nodes.
  - Receives desired-state changes via the Cluster and applies them in the runner thread.

- **NodeRuntime**  
  A per-node manager that:

  - Owns the node’s **SimProcs** and all its **EventQueues**.
  - Provides methods to send/receive events to the node.
  - Gives attention to the node's SimProcs in the right order.
  - Keeps track of the active SimProc

- **Node**
  A unit of simulation logic (e.g. a state machine, step function, or handler) that:
  - Receives events from other nodes and processes them in various simulation processes.
  - Submits new events to other nodes and simulation processes.
  - Node implementations are not part of the Disco engine. Only an abstract Node base class is provided that needs to 
    be overriden by specific implementations as part of a Model.  

- **SimProc**  
    A *simulation process*—a distinct simulation timeline handling events for a node. Conceptually:

  - Each node has exactly one SimProc per Graph layer. The ordered list `model.yml: simprocs` defines the SimProc 
  names in layer order. Layer id ↔ simproc order: `layer_id == simproc_index` (and target_simproc is the name at that index).
  - Drains its own EventQueue when the NodeRuntime gives it attention in the runner loop.
  - SimProcs are not directly visible to the Node; they receive new events from the Node through the NodeRuntime.
  - Simprocs are not directly visible to the routing/transport layer; they are addressed through the NodeRuntime.

The precise mapping between graph layers, nodes, and simprocs is described in a later chapter, but the basic idea is: **the Graph defines the layered structure, simprocs implement the behavior per layer.**

### 2.2 Addressing and Targets

- **Target node name**  
  A unique logical name for a node within an experiment/replication (e.g. `"factory_1"`, `"warehouse_A"`) 
  where `"self"` means “this NodeRuntime’s node”. NodeRuntimes resolve `"self"` to their own `node_name` before routing.

- **Target simproc name**  
  A logical name identifying a SimProc within a node (e.g. `"arrival"`, `"processing"`).

### 2.3 Events, Promises, and EventQueues

- **Event**  
  A message carrying actual simulation data. Characteristics:

  - Has a simulation time `epoch` (float).
  - Carries an opaque, already serialized payload (`bytes`).
  - Can have optional `headers: dict[str, str]`.

- **Promise**  
  A small, control-oriented message used by the EventQueue layer to maintain ordering and completeness. Characteristics:

  - Identified by `seqnr` (sequence number).
  - Contains `epoch` and `num_events` (how many events belong to this promise).
  - Never carries a payload (`bytes`) and is always small.

- **EventQueue**  
  A per-node/per-simproc queue inside the NodeRuntime that:

  - Accepts both events and promises.
  - Is drained by the SimProc when the NodeRuntime/Worker runner gives it attention.
  - May enforce ordering and completion guarantees based on promises (details in the NodeRuntime chapter).

- **NodeRuntime send API**  
  The NodeRuntime exposes a high-level API for model code to send messages:

  ```python
  def send_event(
      self,
      target: str,                # "<node>/<simproc>" or "self/<simproc>"
      epoch: float,
      data: Any,
      headers: dict[str, str] | None = None,
  ) -> None: ...

  def send_promise(
      self,
      target: str,                # "<node>/<simproc>" or "self/<simproc>"
      seqnr: int,
      epoch: float,
      num_events: int,
  ) -> None: ...
  ```

  The NodeRuntime is responsible for:

  - Resolving `"self"` into a concrete node name.
  - Delivering incoming event/promise to the EventQueue of the SimProc identified by envelope.target_simproc.
  - Routing new events from the Node via the active SimProc.
  - Constructing appropriate **envelopes** and delegating to the Router.

### 2.4 Envelopes, Routing, and Transports

- **Envelope**  
  An immutable object used to carry events and promises across process or machine boundaries:

  - `EventEnvelope` – node, simproc, epoch, serialized bytes, headers.
  - `PromiseEnvelope` – node, simproc, seqnr, epoch, num_events.

  Envelopes are used by **transports**; model code and simprocs typically see only higher-level events and promises.

- **Router**  
  A per-Worker routing component that:

  - Knows the current replication id (`repid`).
  - Consults the **Cluster’s address book** to understand where nodes live.
  - Holds an ordered list of **transports** and chooses the first one whose `handles_node(repid, node)` returns `True`.
  - Delegates `send_event` and `send_promise` to the chosen transport.

- **Transport**  
  A concrete mechanism for delivering envelopes between processes/machines. All transports implement a common interface:

  - `handles_node(repid, node) -> bool`
  - `send_event(repid, envelope)`
  - `send_promise(repid, envelope)`

  Current transport types:

  - **InProcessTransport** – direct function calls into local NodeRuntimes.
  - **IPCTransport** – uses `multiprocessing.Queue` and shared memory between processes on the same machine.
  - **GrpcTransport** – uses gRPC (with protobuf messages) to communicate with Workers on other machines, with prioritized, retried promise delivery.

### 2.5 Metastore and Cluster (Control Plane)

- **Metastore**  
  A high-level abstraction over ZooKeeper providing:

  - Hierarchical key–value storage (`update_key`, `get_key`, `update_keys`, `get_keys`).
  - Logical path namespacing (groups).
  - Watch callbacks for changes.
  - Optional queue primitives.

  Metastore is **process-local**: each application process owns its own Metastore instance and ZooKeeper connection manager.

- **Cluster**  
  A higher-level view built on top of the Metastore. It:

  - Tracks **registered workers** and their ephemeral state.
  - Stores per-worker metadata (“worker info”), such as:
    - Application id (`uuid4`) for grouping Workers in the same application.
    - Worker process address (for IPC/gRPC).
    - NUMA node for affinity-aware scheduling.
    - Node assignments, experiment/replication/partition ids.
  - Maintains an **address book** mapping `(repid, node)` → worker address.
  - Exposes a **desired-state** mechanism:
    - Desired state is stored as a **single JSON blob per worker**, so Workers never see partial updates.
    - The Worker subscribes via `on_desired_state_change(worker_address, handler)`.
    - Handlers run in a different thread and signal the Worker runner via conditions.

  Cluster itself is agnostic of local Worker internals; it only manipulates metadata and desired state.

### 2.6 WorkerState and Ingress

- **WorkerState**  
  An `IntEnum` describing the lifecycle of a Worker:

  ```python
  class WorkerState(IntEnum):
      CREATED = 0
      AVAILABLE = 1
      INITIALIZING = 2
      READY = 3
      ACTIVE = 4
      PAUSED = 5
      TERMINATED = 6
      BROKEN = 9
  ```

- **Ingress rules**  
  Transports and gRPC/IPC receivers obey WorkerState:

  - Ingress accepted for:
    - `READY`
    - `ACTIVE`
    - `PAUSED` (queues may fill; backpressure applies)
  - Ingress rejected or failed for:
    - `CREATED`
    - `AVAILABLE`
    - `INITIALIZING`
    - `TERMINATED`
    - `BROKEN`

  Delivery failures (especially for promises) are logged; unrecoverable failures cause the Worker to transition to `BROKEN`.

### 2.7 Graphs, Scenarios, and Masks (Data + Structure Layer)

- **Scenario**  
  A named graph instance stored in the database. Scenarios model the **structural aspects of a simulation** (e.g. supply chain networks, process graphs, resource graphs) and are stored in the dedicated `graph` schema.

- **Graph**  
  The main in-memory structure (`disco.graph.core.Graph`) representing:

  - Vertices and layered edges (one GraphBLAS matrix per layer).
  - Labels and label types (as matrices/vectors).
  - An optional vertex mask (`GraphMask`) used to select subgraphs.

  The **Graph provides the basic structure of a simulation scenario**:

  - It defines which entities exist and how they are connected.
  - It describes **layers of interaction** (e.g. physical flows, information flows, capacity layers).
  - It **governs how nodes will be created and how they interact** at runtime: nodes and their relationships are derived from the graph structure when experiments are loaded.
  - **Simprocs are closely linked to the graph layers**: a typical pattern is that each layer corresponds to a simproc that represents the timeline for that layer.

  The exact mapping (vertex ↔ node, layer ↔ simproc, label ↔ configuration) is defined in later chapters, but this spec assumes that **the graph is the canonical structural model** for a scenario.

- **GraphMask**  
  A wrapper around a `Vector[BOOL]`:

  - Represents a selection of vertices (a subgraph).
  - Can be persisted temporarily in `graph.vertex_masks` using a UUID.
  - Enables efficient DB queries by joining on the mask instead of sending large `IN` lists.

The graph/scenario subsystem is orthogonal to the runtime in implementation, but conceptually it is the **source of truth for simulation structure**: runtime components (Workers, NodeRuntimes, simprocs) are configured from graph and scenario metadata rather than ad-hoc configuration.

### 2.8 Configuration and Settings

- **Application configuration**  
  Disco uses a config module (e.g. `config.py`) with Pydantic models to configure:

  - Logging
  - Database (SQLAlchemy engine)
  - Metastore / ZooKeeper connection
  - gRPC settings (bind address, timeouts, keepalive, compression, retry policies)

- **GrpcSettings** (excerpt)  
  Includes fields such as:

  - `bind_host`, `bind_port`, `timeout_s`, `max_workers`, `grace_s`
  - Message size limits and keepalive options
  - `compression`
  - Promise retry configuration:
    - `promise_retry_delays_s` (backoff sequence, e.g. `[0.05, 0.15, 0.5, 1.0, 2.0]`)
    - `promise_retry_max_window_s` (e.g. `3.0` seconds)

These settings are consumed by the gRPC server and `GrpcTransport` to ensure consistent behavior across applications and environments.

## 3 Metadata Store (Zookeeper‑Backed)

### Purpose
The Metadata Store provides a distributed, fault‑tolerant key–value hierarchy used by simulation servers to coordinate state, assignments, routing metadata, replication information, and other dynamic configuration elements. It acts as the system-wide source of truth for metadata that must be shared across processes, nodes, or simulation clusters.

### Design Overview
The metastore is implemented as a thin, high‑level API on top of ZooKeeper via the `ZkConnectionManager`. It provides atomic hierarchical operations, optional path namespacing via “groups”, lightweight pub/sub via watch callbacks, and structured data expansion semantics for nested trees.

The metastore uses:
- **ZkConnectionManager**: one ZooKeeper client per process, automatically reconnected, watchers restored.
- **Serialization**: pluggable `packb` / `unpackb` (defaults: `pickle.dumps` / `pickle.loads`).
- **Hierarchical API**: `update_key`, `update_keys`, `get_key`, `get_keys`.
- **Optimistic concurrency**: `VersionToken` + compare-and-set updates and single-key atomic update loops.
- **Expand semantics**: declarative control over nested tree retrieval and persistence.
- **Watch support**: callback functions attached to paths, automatically restored after reconnects.

### Path Handling & Namespacing
The metastore exposes *logical paths* (e.g. `"replications/r1/assignments/a"`). These are mapped to ZooKeeper paths via:

```
_full_path(path) =
    "/" + group + "/" + path        if group is set
    "/" + path                      otherwise
```

Examples:
- group = `sim1`, path = `"foo/bar"` → `"/sim1/foo/bar"`
- group = None, path = `"foo/bar"` → `"/foo/bar"`

This allows multiple simulations or tenants to share the same ZK cluster.

### Base Structure
On initialization, the metastore ensures a predefined directory structure (`BASE_STRUCTURE`) inside the chroot. If a group is configured, these structures are created inside the group namespace.

### Serialization
All stored values are serialized using:
- `packb(value) -> bytes`
- `unpackb(bytes) -> value`

Users may inject custom serializers (e.g. msgpack, raw JSON, custom binary formats).

### Watch Callbacks
Watchers are registered through:

```
watch_with_callback(path, callback)
```

Callbacks receive `(value, full_path)` and must return:
- **True** → continue watching  
- **False** → unregister the watcher  

Deletion events pass `raw=None` to the wrapper, which stops the watch automatically.

The `ZkConnectionManager` restores all watches on reconnect.

### Key–Value Operations

#### `update_key(path, value)`
Stores a single leaf value.

#### `get_key(path)`
Reads and deserializes a leaf value. Returns `None` if not present.

#### `get_key_with_version(path)`
Reads and deserializes a leaf value and also returns a `VersionToken` that represents the node’s current version.
If the key does not exist, returns `(None, None)`.

#### `update_key(path, value, expected=VersionToken)`
Performs a **compare-and-set (CAS)** update when `expected` is provided.
The write succeeds only if the stored version matches `expected.value`; otherwise ZooKeeper raises `BadVersionError`.

#### `compare_and_set_key(path, value, expected)`
CAS helper that returns `True` if the update succeeded, and `False` if the key was missing or concurrently modified.

#### `atomic_update_key(path, updater, ...)`
Atomic read–modify–write for a **single key** using an optimistic CAS loop:

- Read `(current_value, version)` via `get_key_with_version()`.
- Compute `new_value = updater(current_value)`.
- Attempt CAS write via `compare_and_set_key()`.
- Retry with exponential backoff + jitter on contention.

If `create_if_missing=True`, the key is created with `updater(None)` if it does not yet exist (racing creators are handled).


### Concurrency, Atomicity, and Multi‑Process Updates
Workers running in different OS processes may update the same logical objects in the metastore concurrently.
The metastore therefore supports **optimistic concurrency control** for single keys via `VersionToken`, CAS updates, and `atomic_update_key()`.

**Transaction boundary:** ZooKeeper guarantees atomicity per znode, not per subtree. Therefore, objects that must be updated atomically
(e.g. `Experiment` state updates) should be stored as a **single value under one key** (one znode).

**Recommended pattern (CAS loop):**
- Read the current object blob (e.g. experiment dict) and version.
- Apply the mutation in memory (including status propagation).
- Write back using CAS. On contention, retry.

This pattern provides a single‑transaction update in ZooKeeper today and maps directly to etcd transactions later (compare on `mod_revision`).

**Failure semantics:** `atomic_update_key()` raises `MetastoreConflictError` after repeated contention; callers may retry at a higher level or fail fast.
### Hierarchical Get/Update

#### Expand Semantics
The `expand` parameter dictates how nested structures are written or read:

- **`expand = {"replications": {"assignments": None}}`**  
  → fully expand `replications`, expand one level for `assignments`.

- **`expand = {"replications": None}`**  
  → expand `replications` only one level; store subtrees as dictionaries.

Examples:

With:
```
members = {"replications": {"r1": {"assignments": {"a": 1}}}}
```

**Case A — nested expand**
```
expand = {"replications": {"assignments": None}}
```
Writes:
```
/replications/r1/assignments/a = 1
```

**Case B — shallow expand**
```
expand = {"replications": None}
```
Writes:
```
/replications/assignments = {"a": 1}
```

### Automatic Parent Node Behavior
ZooKeeper itself distinguishes between nodes and their children. Our FakeKazooClient mimics this semantics: a parent exists if it has either data or children. This ensures `get_keys()` behaves consistently.

### Queue Operations
The metastore exposes simple FIFO queue operations backed by ZooKeeper’s `Queue` recipe:
- `enqueue(path, value)`
- `dequeue(path, timeout=None)`

These are used for lightweight inter-node message passing or distribution of pending events.

### Failure & Recovery Semantics
- All client operations route through a **single client instance** owned by `ZkConnectionManager`.
- Session loss triggers automatic reconnection and watch reinstallation.
- `update_keys` and `get_keys` operate only on logical paths, ensuring compatibility with grouping and chrooting.

### Intended Usage in the Application
- Store routing tables, node status, simulation assignments, replication metadata.
- Provide shared configuration across long‑lived processes.
- Support live reconfiguration without restarts.
- Enable efficient, fine-grained read access to metadata subsets.
- Allow stateless workers to bootstrap by reading the full hierarchical metadata tree.

### Limitations / Non-Goals
- Not designed for large binary payloads (those must go to shared memory or the data layer).
- Not a transactional database—ZooKeeper operations are atomic per node, not per subtree. Single‑key atomic updates are supported via CAS and `atomic_update_key()`.
- Not a metrics store or event log.

### Testing Requirements
- Use `FakeKazooClient` and `FakeConnectionManager` for isolated testing.
- Must test:
  - expand semantics round-trip (write → read)
  - watch registration and deletion behavior
  - recoverability after reconnection
  - queue timeouts and ordering
  - group namespacing in paths
  - handling of scalar vs dictionary values in expansion
  - versioned reads (`get_key_with_version`) and version increments on update
  - CAS behavior for `update_key(..., expected=...)` (success + `BadVersionError` on stale token)
  - `compare_and_set_key` return semantics on success, contention, and missing node
  - `atomic_update_key` create/update paths, retry-on-contention behavior, and `MetastoreConflictError` after `max_retries`


## 4 Cluster

### 4.1 Purpose

The **Cluster** component provides a high-level, read-mostly, strictly in-process representation of the current simulation topology. It is built entirely on top of the `Metastore` (Chapter 3), which abstracts all ZooKeeper behavior, connection management, and watcher restoration.

Cluster has no knowledge of Worker internals. It never interacts with Worker objects, NodeRuntimes, transports, or runtime execution. Its sole responsibility is to interpret metadata exposed in the Metastore and present a coherent view of:

- Which workers exist,
- Which nodes they host,
- Their experiment (`expid`) and replication (`repid`), both UUID4,
- Their application and NUMA grouping,
- Their runtime state,
- How to route messages to them.

This metadata is consumed by the transport layer and by the Worker lifecycle controller.

### 4.2 Worker Metadata (Persistent)

Each Worker process publishes structured metadata in the Metastore under:

```
/simulation/workers/{worker_address}/
```

All of these keys are written atomically using `Metastore.update_keys()`.

#### Metadata fields

| Key | Type | Description |
|-----|-------|-------------|
| `expid` | UUID4 string | Experiment ID to which this worker belongs |
| `repid` | UUID4 string | Replication ID for this worker |
| `partition` | int | Partition index within the replication |
| `nodes` | list[str] | Names of nodes hosted on this worker |
| `application_id` | UUID4 string | Identifier of the application process that spawned the worker |
| `numa_node` | int | NUMA node on which the worker runs |

Cluster treats this metadata as definitive topology information.

#### Malformed or incomplete metadata

If Cluster encounters missing, malformed, or nonsensical metadata:

**The corresponding worker is considered BROKEN.**

Cluster does not attempt partial interpretation or correction. The orchestration layer must recreate the worker.

### 4.3 Worker WorkerState (Ephemeral)

Each worker publishes its runtime state in an ephemeral key:

```
/simulation/registered_workers/{worker_address}
```

The value is an integer representing `WorkerState`:

```
0 CREATED
1 AVAILABLE
2 INITIALIZING
3 READY
4 ACTIVE
5 PAUSED
6 TERMINATED
9 BROKEN
```

Because the node is ephemeral:

- If the worker process dies, the key disappears automatically.
- Cluster removes the worker and all associated routing information.

Cluster does **not** perform any state transitions; it only reflects what the worker publishes.

### 4.4 Desired WorkerState (Control Plane Input)

Each worker receives its desired operational state via a **single structured value** stored under:

```python
/simulation/desired_state/{worker_address}/desired
```

This value is written atomically by the orchestrator as an object of type:

```python
DesiredWorkerState:
    request_id: UUID4
    expid: UUID4 | None
    repid: UUID4 | None
    partition: int | None
    nodes: list[str] | None
    state: WorkerState
```

**Serialization and deserialization are fully handled by the Metastore.**

Cluster neither serializes nor deserializes these values—it receives and emits Python objects exactly as returned by the 
Metastore’s unpackb function.

#### Cluster Responsibilities

Cluster does **not** interpret the fields inside the desired state. Its responsibilities are limited to:

- Installing a watch on the worker’s desired-state path.
- Receiving decoded values from the Metastore upon change.
- Delivering them to subscriber code via:

```python
cluster.on_desired_state_change(worker_address, handler)
```

The handler signature is:

```python
handler(desired_state: DesiredWorkerState) -> str | None
```

Return semantics:

- `None` → request accepted successfully  
- `str` → error message; request considered failed

Cluster writes the acknowledgment to:

```python
/simulation/desired_state/{worker_address}/ack
```

Acknowledgment structure:

```python
{
    request_id: <same as input>,
    success: bool,
    error: str | None
}
```

#### Subscription Model

Cluster does **not** maintain any internal list of subscribers.

Each call to:

```python
on_desired_state_change(worker_address, handler)
```

installs exactly one watcher via Metastore, and the handler is invoked for every update until the watch is removed 
(e.g., deletion of the node or callback returning False).

### 4.5 Address Book

From valid worker metadata, Cluster derives a mapping:

```python
address_book: Mapping[tuple[str, str], str]
# (repid: UUID4, node_name: str) -> worker_address: str
```

Semantics:

- For each `(repid, node)`, the address identifies where the node is hosted.
- This address may serve multiple nodes or partitions.
- UUID4 `repid` ensures globally unique replication identifiers.

Cluster also exposes helper classification functions:

```python
is_local_address(worker_address: str, application_id: str) -> bool
is_ipc_reachable(worker_address: str, application_id: str, numa_node: int) -> bool
is_remote_address(worker_address: str) -> bool
```

These guides transport selection but do not perform routing themselves.

### 4.6 Internal Data Structures

Cluster maintains synchronized in-memory structures:

- `worker_meta: dict[worker_address, WorkerInfo]`
- `worker_state: dict[worker_address, WorkerState]`
- `desired_state: dict[worker_address, dict]`
- `address_book: Mapping[(repid, node), worker_address]`
- `application_groups: dict[application_id, set[worker_address]]`
- `numa_layout: dict[worker_address, int]`

All modifications occur under a Cluster-level lock to guarantee consistency.

Watch callbacks from Metastore update only the affected sections.

### 4.7 Error Model and Recovery

#### Metastore disconnection

Handled **entirely** by Metastore. Cluster:

- Does not manage watchers,
- Does not manage reconnection,
- Does not implement failover logic.

When the Metastore reconnects, Cluster naturally receives updated callbacks and rebuilds state.

#### Malformed metadata

If a worker publishes invalid metadata:

- Cluster treats it as a fatal condition for that worker.
- The worker must be recreated by orchestration.
- Cluster removes the worker from routing.

#### Worker removal

If the ephemeral key disappears:

- The worker is immediately removed.
- All routing entries for nodes it hosted are dropped.

Cluster never attempts to reassign or recover nodes.

### 4.8 Non-Responsibilities

Cluster explicitly does **not**:

- Infer the local worker's address,
- Interact with Workers or NodeRuntimes,
- Perform routing or transport selection,
- Modify or validate desired-state semantics,
- Move workers through the state machine,
- Handle Metastore recovery mechanisms.

It is a pure metadata reflector, translating the contents of the Metastore into in-memory structures for other components to consume.

## 5 Worker Architecture & Lifecycle

### 5.1 Overview

A Worker is a long-lived simulation process responsible for hosting a set of
nodes (as `NodeRuntime` instances) and executing their logic for exactly one
**assignment** at a time: `(expid, repid, partition)`.

The Worker is driven by **DesiredWorkerState** updates published via the Cluster.
All state transitions and all simulation stepping happen on the Worker's runner
thread (the main thread of the worker process) for determinism and to avoid
concurrent mutation of runtimes.

The Worker separates responsibilities into:

- **Control plane (slow path):** apply desired-state updates, setup/teardown runs,
  publish WorkerState, and update experiment/partition status.
- **Data plane (hot path):** drain ingress queues and step NodeRuntime runners.

### 5.2 Worker Responsibilities

A Worker is responsible for:

- Hosting `NodeRuntime` instances for all nodes assigned to its partition.
- Running a deterministic stepping loop during `ACTIVE`.
- Maintaining and publishing `WorkerState` to the Cluster.
- Hosting transports and routing for intra-worker and inter-worker messaging.
- Draining ingress (IPC/gRPC) and delivering events/promises to NodeRuntimes.
- Reacting to Cluster desired-state changes.

Workers do not perform low-level ZooKeeper operations directly. They interact
with experiment metadata through the `ExperimentStore` abstraction (load
experiment data, update per-partition status, report per-partition exceptions).

### 5.3 WorkerState Machine

The WorkerState enum is:

- `CREATED`
- `AVAILABLE`
- `INITIALIZING`
- `READY`
- `ACTIVE`
- `PAUSED`
- `TERMINATED`
- `EXITED`
- `BROKEN`

Definitions:

- **CREATED** -- process started and registered, not yet available for work.
- **AVAILABLE** -- idle and ready to accept a new assignment.
- **INITIALIZING** -- preparing a specific assignment (loading inputs, creating NodeRuntimes).
- **READY** -- run is initialized; waiting for command to start executing.
- **ACTIVE** -- simulation executing; NodeRuntimes are stepped repeatedly.
- **PAUSED** -- simulation not stepped; run state preserved.
- **TERMINATED** -- abort requested; run is torn down and worker returns to AVAILABLE.
- **EXITED** -- shutdown requested; worker stops its runner loop and exits.
- **BROKEN** -- unrecoverable internal error; worker must be restarted.

Workers do **not** have FINISHED/FAILED worker states. Completion/failure is
tracked at the experiment/partition level (ExperimentStatus), after which the
Worker returns to `AVAILABLE`.

### 5.4 Interaction With NodeRuntimes

Each assigned node is represented by a `NodeRuntime`. NodeRuntimes own:

- SimProc instances (one per layer / simproc definition)
- Event/promise intake queues (conceptually per SimProc)
- Node-specific simulation logic

The Worker:

- Creates NodeRuntimes during `INITIALIZING` (part of the READY transition).
- Configures routing/transports (once per worker lifetime; reused across runs).
- Delivers ingress messages (events/promises) into NodeRuntimes.
- Steps NodeRuntimes in a stable order during `ACTIVE`.

### 5.5 Receiving Desired-WorkerState Changes

The Cluster delivers desired-state updates via:

- `cluster.on_desired_state_change(worker_address, handler)`

The handler is invoked on a separate callback thread. It must:

- Store the received desired-state blob in a thread-safe slot (latest value wins).
- Wake the runner thread.

Implementation detail: the Worker uses a `threading.Event` (e.g. `_kick`) to wake
the runner thread. The event is set on desired-state updates and on `request_stop`.
Ingress messages do **not** set `_kick` (by design) to avoid per-message overhead.

### 5.6 Runner Loop

The runner loop keeps the `ACTIVE` hot path extremely small and fast, while still
reacting promptly to control-plane updates.

Key ideas:

- A **control tick** runs at most once per second (or earlier if kicked).
- The **ACTIVE hot path** does no locking in the common case: drain ingress then step runners.
- Locks are taken only for: applying desired-state, reporting status, teardown, and errors.

High-level pseudocode (not exact code):

- Every loop:
  - If kicked or 1s elapsed: apply pending desired-state (under lock).
  - If `ACTIVE`: drain ingress; step runners.
  - If not `ACTIVE`: wait until kick or next control tick.

### 5.7 Applying Desired-WorkerState Commands

Desired-state commands include an assignment and a target WorkerState.

Core invariants:

1. **Assignment is set only on transition to `READY`.**
2. **Runners are created only on `READY -> ACTIVE`.**
3. **During `ACTIVE`, the runner loop never calls setup/teardown/control helpers.**
4. **Pending desired-state updates overwrite earlier ones (latest wins).**

State transitions (summary):

- `READY`: validate, set assignment, enter `INITIALIZING`, run setup, publish `READY`.
- `ACTIVE`: if coming from `READY`, create runners once; publish `ACTIVE` and set partition status ACTIVE.
- `PAUSED`: publish `PAUSED` and set partition status PAUSED.
- `TERMINATED`: abort run, set partition status to CANCELED (when applicable), teardown, return to `AVAILABLE`.
- `EXITED`: teardown and stop the runner loop.

### 5.8 ExperimentStatus Lifecycle for a Partition

Experiment/partition status is updated through `ExperimentStore` helpers:

- `set_partition_status(expid, repid, partition, status)`
- `set_partition_exc(expid, repid, partition, exc, fail_partition=True)`

Typical partition lifecycle (happy path):

- `LOADED` -- prerequisites loaded (experiment, graph, partitioning, model).
- `INITIALIZED` -- NodeRuntimes created and `initialize(**params)` completed.
- `ACTIVE` -- worker enters `ACTIVE` and runners are created.
- `FINISHED` -- all NodeRuntime runners completed and the run ends.

Interruptions:

- `PAUSED` -- controller requested pause.
- `CANCELED` -- controller requested terminate before completion.
- `FAILED` -- setup or stepping raised a partition-fatal exception, reported via `set_partition_exc(..., fail_partition=True)`.

Classification rule of thumb:

- Fail the **partition** when the error is specific to experiment/scenario/partition inputs.
- Mark the **worker BROKEN** when the process cannot function reliably (e.g. cannot load the fixed model package, cannot report status to metastore, internal invariants violated).

### 5.9 Ingress and Backpressure

Ingress messages arrive via IPC and gRPC and are queued for the Worker.

- In `ACTIVE`, ingress is drained continuously (each hot-path cycle).
- In `READY` / `PAUSED`, ingress may be drained opportunistically (e.g. on control ticks), but timely delivery is not required for correctness because the simulation is not stepping.
- In states where the run is not configured (`CREATED`, `AVAILABLE`, `INITIALIZING`, `TERMINATED`, `EXITED`, `BROKEN`), ingress is effectively rejected or ignored.

Backpressure is implemented by transports and queue limits.

### 5.10 Error Handling

Two distinct outcomes exist:

- **Partition failure (worker stays healthy):** report partition exception (and mark FAILED), teardown, return to `AVAILABLE`.
- **Worker failure (BROKEN):** publish `BROKEN` to the Cluster and stop; recovery requires restarting the worker process.

### 5.11 Summary

- Worker execution is deterministic and single-threaded.
- Desired-state updates are received asynchronously but applied on a slow control tick (<= 1 Hz) or on a kick.
- The `ACTIVE` hot path is minimized: drain ingress and step runners without locks.
- ExperimentStatus is maintained per partition via ExperimentStore, not WorkerState.
- Errors are classified into partition-fatal (fail partition) and worker-fatal (BROKEN).

## 6 Routing and Transports

### 6.1 Purpose and Scope

This chapter describes how messages (events and promises) are routed between
nodes and workers in a simulation cluster. It covers:

- The **Router**, which chooses a transport for each outgoing message.
- The abstract **Transport** interface.
- Concrete transports:
  - **InProcessTransport** (same-process delivery via `NodeRuntime`).
  - **IPCTransport** (inter-process communication via queues + shared memory).
  - **GrpcTransport** (remote communication between workers over gRPC).
- The **gRPC ingress** service that receives envelopes from remote workers and
  injects them into the local IPC queues.

The goal is to provide a layered, extensible routing architecture where:

- Local messages are delivered as cheaply as possible.
- Intra-host messages use efficient IPC channels.
- Cross-host messages use gRPC with clear retry semantics.
- The routing policy is determined by a single place (the `Router`),
  not scattered throughout the codebase.


### 6.2 Addressing and Locality Model

The routing subsystem provides a uniform way for simulation components to send *envelopes* (events and promises) between nodes, regardless of whether the destination node is local (same process) or remote (different worker process).

Key ideas:

- The **Router** is a lightweight dispatcher that owns an ordered list of **Transports**.
- A **Transport** implements one delivery mechanism (in-process calls, IPC, gRPC, …) and encapsulates any dependencies needed to deliver (queues, sockets, addressing helpers, etc.).
- Local delivery must be fast: `InProcessTransport` uses a shared mapping `{node_name -> NodeRuntime}` and calls `NodeRuntime.receive_event(...)` / `receive_promise(...)` directly.
- The Router itself does not require a Cluster. In the distributed Worker architecture, *remote* transports may still depend on Cluster-provided metadata (e.g. where a node currently lives), but that dependency is contained within those transports.

This separation is what allows lightweight contexts (like `TestRun`) to use `Router + InProcessTransport` without requiring a Cluster or Metastore.

### 6.3 Router

The **Router** is a worker-local component responsible for selecting a
transport for each outgoing envelope. It is **long-lived**: a worker creates
its router once and reuses it for all experiments and replications. The
router does not store a specific `repid` in its constructor; instead, the
current `repid` is carried by the envelopes themselves.

#### 6.3.1 Responsibilities

- Own an ordered list of `Transport` instances.
- For each outgoing `EventEnvelope` or `PromiseEnvelope`:
  - Read `repid` from the envelope.
  - Ask each transport in order whether it **handles** the target node for
    that `repid`.
  - Use the first transport that responds positively.
  - Raise an error if no transport claims responsibility.

The priority order is determined by the worker during construction, typically:

1. `InProcessTransport`
2. `IPCTransport`
3. `GrpcTransport`

This ensures that the cheapest delivery mechanism is chosen when multiple
transports could technically reach the same node.

#### 6.3.2 Construction

`Router` is constructed from a list of transport instances. The Router itself is intentionally “dumb”: it does not talk to the Cluster or Metastore, and it does not own any routing state beyond the transport list.

Any dependencies needed for addressing (for example: resolving a remote worker address for a node) live inside the relevant transport implementation, not inside the Router. This keeps the Router easy to test and makes it usable in single-process contexts.

In pseudocode (illustrative only):

- `inproc = InProcessTransport(nodes=<mapping node_name -> NodeRuntime>)`
- `router = Router(transports=[inproc, ...])`

#### 6.3.3 Routing Logic

At runtime, the Router tries to deliver an envelope using the first transport that can accept it. The core rule is locality:

- If the target node is hosted *in the current process*, the Router delivers via `InProcessTransport` (direct function calls into the target `NodeRuntime`).
- Otherwise, the Router attempts one of the remote transports (e.g. IPC, gRPC). Remote transports are responsible for knowing *how* to reach the destination (worker address resolution, connection management, serialization, retries, backpressure, etc.).

This separation keeps the hot path small: “local delivery” should be a cheap dictionary lookup plus a direct call, with no Cluster interaction.

#### 6.3.4 Introspection

The router exposes helpers:

- `transports() -> Iterable[Transport]` — underlying transports in priority order.
- `transport_names() -> list[str]` — names of transports in priority order.

These are mainly for diagnostics, tests, and debugging tools.


### 6.4 Transport Interface

All transports implement a common protocol (defined in `transports.base`):

```python
class Transport(Protocol):
    def handles_node(self, repid: str, node: str) -> bool: ...
    def send_event(self, envelope: EventEnvelope) -> None: ...
    def send_promise(self, envelope: PromiseEnvelope) -> None: ...
```

Key points:

- `handles_node` takes both `repid` and `node`, since routing may depend on
  which replication a node belongs to.
- `send_event` and `send_promise` **do not** receive `repid` as a separate
  parameter. Instead, all transports read `envelope.repid` when they need it.
- Transports are **long-lived**: a worker constructs its transports once
  (e.g. during startup) and reuses them for multiple runs and replications.

Semantics:

- `handles_node` must be **pure and fast**: it should not perform blocking I/O.
  It is allowed to consult in-memory data (e.g. the address book or local maps).
- `send_event` and `send_promise` may perform I/O and may raise exceptions on
  failure. Exceptions propagate up to the caller (typically `NodeRuntime`),
  which can decide whether to log, retry, or treat the failure as fatal.

Transport implementations must avoid **double serialization** of payloads.
Payloads are serialized exactly once in `NodeRuntime` before envelopes are
handed to the router.


### 6.5 InProcessTransport

The **InProcessTransport** delivers messages directly to `NodeRuntime`
instances in the same process.

#### 6.5.1 Construction

```python
@dataclass(slots=True)
class InProcessTransport(Transport):
    nodes: Mapping[str, NodeRuntime]
    cluster: Cluster
```

- `nodes` maps node names to their local `NodeRuntime` instances.
- `cluster` is used only for address-book checks in `handles_node`.

This transport is also long-lived. The worker owns a single `InProcessTransport`
instance and updates its `nodes` mapping as it creates and tears down
`NodeRuntime`s across runs.

#### 6.5.2 Routing Decision

```python
def handles_node(self, repid: str, node: str) -> bool:
  if node not in self.node_specs:
    return False
  return (repid, node) in self.cluster.address_book
```

A node is considered in-process if:

- The worker hosts a `NodeRuntime` for that node, and
- The address book has an entry for `(repid, node)` mapping to this worker's
  address (ensured by `Cluster` and `Worker` assignment logic).

#### 6.5.3 Delivery

Events:

```python
def send_event(self, envelope: EventEnvelope) -> None:
  node = self.node_specs[envelope.target_node]
  node.receive_event(envelope)
```

Promises:

```python
def send_promise(self, envelope: PromiseEnvelope) -> None:
  node = self.node_specs[envelope.target_node]
  node.receive_promise(envelope)
```

`NodeRuntime.receive_event` and `receive_promise` are responsible for
pushing envelopes into their internal queues and integrating them into the
deterministic runner loop controlled by the `Worker`.


### 6.6 IPC Transport (Queues + Shared Memory)

The **IPCTransport** supports communication between processes on the same host
(or within a tightly coupled environment) via `multiprocessing.Queue` and
optional shared memory for large payloads.

#### 6.6.1 Egress (IPCTransport)

```python
class IPCTransport(Transport):
    def __init__(
        self,
        cluster: Cluster,
        event_queues: Mapping[str, Queue[IPCEventMsg]],
        promise_queues: Mapping[str, Queue[IPCPromiseMsg]],
        large_payload_threshold: int = 64 * 1024,
    ) -> None:
        ...
```

- `cluster` provides the address book mapping `(repid, node)` to a worker address.
- `event_queues` maps worker addresses to event queues.
- `promise_queues` maps worker addresses to promise queues.
- `large_payload_threshold` determines when to spill payloads into shared
  memory instead of putting them inline on the queue.

Routing decision:

```python
def handles_node(self, repid: str, node: str) -> bool:
    addr = self._cluster.address_book.get((repid, node))
    if addr is None:
        return False
    return addr in self._event_queues and addr in self._promise_queues
```

A node is routable via IPC if both an event queue and a promise queue are
available for its address.

Event send:

- Read `repid` from `envelope.repid`.
- Resolve `addr = address_book[(repid, envelope.target_node)]`.
- If `len(envelope.data) <= large_payload_threshold`:
  - Construct `IPCEventMsg` with `repid`, inline `data`, and `shm_name=None`.
  - Put the message on `event_queues[addr]`.
- Else:
  - Allocate a `SharedMemory` block of size `len(envelope.data)`.
  - Copy `envelope.data` into the buffer.
  - Construct `IPCEventMsg` with `repid`, `data=None`, `shm_name` set to the
    shared memory name, and `size` set to the payload length.
  - Put the message on the event queue.

Promise send:

- Read `repid` from `envelope.repid`.
- Resolve `addr = address_book[(repid, envelope.target_node)]`.
- Construct `IPCPromiseMsg` with `repid` and the promise metadata.
- Put the message on `promise_queues[addr]`.


#### 6.6.2 IPC Message Types

`IPCEventMsg` and `IPCPromiseMsg` explicitly carry `repid`, so that receivers
and workers do not need out-of-band information to know which replication a
message belongs to.

```python
@dataclass(slots=True)
class IPCEventMsg:
    repid: str
    target_node: str
    target_simproc: str
    epoch: float
    headers: Dict[str, str]
    data: Optional[bytes]
    shm_name: Optional[str]
    size: int

@dataclass(slots=True)
class IPCPromiseMsg:
    repid: str
    target_node: str
    target_simproc: str
    seqnr: int
    epoch: float
    num_events: int
```
- For small payloads, `data` is the serialized bytes and `shm_name` is `None`.
- For large payloads, `data` is `None`, `shm_name` is the name of the shared
  memory handle, and `size` is the actual payload length.


#### 6.6.3 IPCReceiver

`IPCReceiver` provides a small helper for reading from IPC queues and
forwarding envelopes to local `NodeRuntime`s.

```python
class IPCReceiver:
    def __init__(
        self,
        nodes: Mapping[str, NodeRuntime],
        event_queue: Queue[IPCEventMsg],
        promise_queue: Queue[IPCPromiseMsg],
    ) -> None:
        ...
```
Responsibilities:

- `run_event_loop()` — blocking loop: consume `IPCEventMsg` from `event_queue`,
  reconstruct `EventEnvelope`, and call `NodeRuntime.receive_event`.
- `run_promise_loop()` — blocking loop: consume `IPCPromiseMsg` from
  `promise_queue`, reconstruct `PromiseEnvelope`, and call
  `NodeRuntime.receive_promise`.
- `_extract_event_data(msg: IPCEventMsg) -> bytes` — internal helper to restore
  the payload from inline data or shared memory. It is responsible for closing
  and unlinking the shared memory segment after reading to avoid leaks.

Note: in the current design, the **Worker** often plays the role of
"IPCReceiver" by draining its own ingress queues inside the runner loop, so
that promises can be prioritized over events.


### 6.7 gRPC Transport (Egress)

The **GrpcTransport** is responsible for sending envelopes to remote workers
over gRPC, using the `DiscoTransport` service defined in
`transports/proto/transport.proto`.

#### 6.7.1 Protobuf Service

The protobuf messages now also carry `repid` explicitly, so that gRPC ingress
can reconstruct full envelopes and IPC messages without external context.

```proto
message EventEnvelopeMsg {
  string repid          = 1;
  string target_node    = 2;
  string target_simproc = 3;
  double epoch          = 4;
  bytes data            = 5;
  map<string, string> headers = 6;
}

message PromiseEnvelopeMsg {
  string repid          = 1;
  string target_node    = 2;
  string target_simproc = 3;
  int64 seqnr           = 4;
  double epoch          = 5;
  int32 num_events      = 6;
}

message TransportAck {
  string message = 1;   // optional diagnostics
}

service DiscoTransport {
  // Events: potentially large and frequent. Use a client-streaming RPC.
  rpc SendEvents(stream EventEnvelopeMsg) returns (TransportAck);

  // Promises: always small. Use a unary RPC for better latency and observability.
  rpc SendPromise(PromiseEnvelopeMsg) returns (TransportAck);
}
```

#### 6.7.2 Construction and Address Resolution

```python
class GrpcTransport(Transport):
    def __init__(
        self,
        cluster: Cluster,
        settings: GrpcSettings,
        channel_factory: Callable[[str, GrpcSettings], grpc.Channel] | None = None,
        stub_factory: Callable[[grpc.Channel], DiscoTransportStub] | None = None,
    ) -> None:
        ...
```

- `cluster` provides the address book.
- `settings: GrpcSettings` holds timeout, compression, and retry parameters.
- `channel_factory` (optional) creates a `grpc.Channel` given a target address
  and settings. In production this defaults to a standard `grpc.insecure_channel`
  with configured options (message size limits, keepalive, compression, ...).
- `stub_factory` (optional) creates a `DiscoTransportStub` from a channel.

Address resolution:

```python
def _resolve_address(self, repid: str, node: str) -> str:
    try:
        return self._cluster.address_book[(repid, node)]
    except KeyError as exc:
        raise RuntimeError(
            f"No address for (repid={repid!r}, node={node!r})"
        ) from exc
```

`handles_node` checks whether `(repid, node)` exists in the address book. The
router ensures that `GrpcTransport` is only used when higher-priority
transports do not apply.

#### 6.7.3 Event Sending

Events are sent via the `SendEvents` client-streaming RPC. For simplicity,
`GrpcTransport.send_event` currently opens a short-lived stream carrying a
single message:

```python
def send_event(self, envelope: EventEnvelope) -> None:
    repid = envelope.repid
    addr = self._resolve_address(repid, envelope.target_node)
    endpoint = self._get_or_create_endpoint(addr)

    msg = transport_pb2.EventEnvelopeMsg(
        repid=repid,
        target_node=envelope.target_node,
        target_simproc=envelope.target_simproc,
        epoch=envelope.epoch,
        data=envelope.data,
        headers=envelope.headers,
    )

    def _iter():
        yield msg

    logger.debug(
        "GrpcTransport.send_event: repid=%s node=%s simproc=%s addr=%s",
        repid,
        envelope.target_node,
        envelope.target_simproc,
        addr,
    )

    endpoint.stub.SendEvents(_iter(), timeout=self._settings.timeout_s)
```
`_get_or_create_endpoint(addr)` returns a cached structure containing the
channel and stub for the given target address. Channels and stubs are reused
across calls to minimize connection overhead.

Errors raised by `SendEvents` propagate back to the caller; there is currently
no retry logic for events (they are expected to be retried at a higher layer if
needed).


#### 6.7.4 Promise Sending with Retry

Promises are sent via the unary `SendPromise` RPC. Because promises are small
and critical for synchronization, `GrpcTransport` implements a retry policy
based on `GrpcSettings`:

- `promise_retry_delays_s: list[float]` — backoff sequence between retry
  attempts (in seconds).
- `promise_retry_max_window_s: float` — maximum time window for retries
  (in seconds). Once this window is exceeded, the last error is surfaced.

Algorithm:

1. Read `repid` from `envelope.repid`.
2. Resolve `addr` via `_resolve_address(repid, envelope.target_node)`.
3. Obtain `_RemoteEndpoint` (channel + stub) for `addr`.
4. Build `PromiseEnvelopeMsg` with all fields, including `repid`.
5. Call `stub.SendPromise(msg, timeout=settings.timeout_s)`.
6. If the call succeeds, return immediately.
7. If the call fails with a retryable error (e.g. `RESOURCE_EXHAUSTED` or
   `UNAVAILABLE`), wait for the next delay in `promise_retry_delays_s`,
   accumulate elapsed time, and retry as long as the total elapsed time is
   below `promise_retry_max_window_s`.
8. Once the retry window is exceeded or a non-retryable error is encountered,
   re-raise the last exception.

This design keeps retry policy centralized and configurable, while allowing the
worker to treat persistent promise delivery failures as higher-level errors
(e.g. marking remote workers as unhealthy).


### 6.8 gRPC Ingress

The **gRPC ingress** is the server-side implementation of `DiscoTransport`
that receives envelopes from remote workers and injects them into the local
IPC queues. The ingress does not store any replication-specific state; it
relies on the `repid` field present in each protobuf message.

#### 6.8.1 DiscoTransportServicer

A typical implementation looks like:

```python
class DiscoTransportServicer(transport_pb2_grpc.DiscoTransportServicer):
    def __init__(
        self,
        event_queue: Queue[IPCEventMsg],
        promise_queue: Queue[IPCPromiseMsg],
    ) -> None:
        self._event_queue = event_queue
        self._promise_queue = promise_queue

    def SendEvents(self, request_iterator, context):
        count = 0
        for msg in request_iterator:
            ipc = IPCEventMsg(
                repid=msg.repid,
                target_node=msg.target_node,
                target_simproc=msg.target_simproc,
                epoch=msg.epoch,
                headers=dict(msg.headers),
                data=msg.data,
                shm_name=None,
                size=len(msg.data),
            )
            self._event_queue.put(ipc)
            count += 1
        return transport_pb2.TransportAck(message=f"Received {count} events")

    def SendPromise(self, request, context):
        ipc = IPCPromiseMsg(
            repid=request.repid,
            target_node=request.target_node,
            target_simproc=request.target_simproc,
            seqnr=request.seqnr,
            epoch=request.epoch,
            num_events=request.num_events,
        )
        self._promise_queue.put(ipc)
        return transport_pb2.TransportAck(message="Promise accepted")
```
Key points:

- The servicer does **not** talk to `NodeRuntime` directly.
- It performs minimal transformation: protobuf messages → IPC message types.
- `repid` is copied verbatim from the protobuf messages into `IPCEventMsg` and
  `IPCPromiseMsg`.
- Ingress acceptance is not gated by worker state here; state gating and
  delivery prioritization are handled by the local worker and its runner loop
  when draining the ingress queues.

#### 6.8.2 Server Bootstrap

A helper function starts the gRPC server for a worker:

```python
def start_grpc_server(
    worker,
    event_queue: Queue[IPCEventMsg],
    promise_queue: Queue[IPCPromiseMsg],
    settings: GrpcSettings,
) -> grpc.Server:
    # Create grpc.Server with ThreadPoolExecutor(max_workers=settings.max_workers)
    # Configure message size limits, keepalive, and compression from GrpcSettings.
    # Register DiscoTransportServicer(event_queue, promise_queue).
    # Bind to worker.address (e.g. "host:port") via server.add_insecure_port.
    # Start and return the server instance.
```
The worker is responsible for:

- Choosing its own `worker.address` (which must match the address stored in the
  `Cluster.address_book` for its nodes).
- Creating the local IPC queues (one event and one promise queue per worker).
- Passing its own queues to `start_grpc_server`.
- Draining those queues in its runner loop, converting `IPCEventMsg` /
  `IPCPromiseMsg` into `EventEnvelope` / `PromiseEnvelope` (using `repid` from
  the message) and delivering them to `NodeRuntime`s.

This design ensures a clean separation:

- `GrpcTransport` (egress) → remote worker ingress via gRPC.
- `DiscoTransportServicer` (ingress) → local IPC queues.
- `Worker` runner loop → deterministic, prioritized delivery into
  `NodeRuntime` instances using the same ingress path for both IPC and gRPC.


## 7. Layered Graph and Scenario Subsystem

This chapter describes the *graph* subpackage: how layered graphs are represented
in memory with python-graphblas, how they are persisted in the relational
`graph.*` schema, and how model data is joined back to the structural graph via
index↔key mappings and vertex masks.

The focus is on four modules:

- `disco.graph.core` — in-memory `Graph` representation (DB‑agnostic).
- `disco.graph.schema` — SQLAlchemy table metadata under the `graph` schema.
- `disco.graph.db` — utilities to create/delete scenarios and to store/load graphs.
- `disco.graph.extract` — helpers to extract model data (Pandas / GraphBLAS) using
  the structural graph plus masks.

`Graph` deliberately does **not** know about SQLAlchemy or the database; the coupling
is done through the `graph.db`, `graph.extract`, and `graph_mask` helpers.


### 7.1 Overview and Responsibilities

At a high level:

- The **structural graph** lives in memory as a `Graph` backed by python‑graphblas:

  - Vertices are numbered `0 .. num_vertices-1` per scenario.
  - Each **layer** is a square adjacency matrix `A_ℓ` of shape
    `num_vertices × num_vertices`.
  - Optional vertex **labels** are stored in a sparse boolean matrix
    (`vertex × label_id`).
  - An optional **mask** is a `Vector[BOOL]` over vertices, wrapped in a
    `GraphMask` to support persistence.

- The **relational schema** under `graph.*` stores structural information and
  the index↔key mapping:

  - `graph.scenarios` holds scenario metadata.
  - `graph.vertices` maps `(scenario_id, index) → key` and anchors structural
    edges and masks.
  - `graph.edges` stores weighted edges per layer in index space.
  - `graph.labels` and `graph.vertex_labels` store label definitions and
    vertex‑label assignments.
  - `graph.vertex_masks` stores persisted vertex masks keyed by a UUID.

- The **model data** (e.g. node tables, edge tables, parameters) is stored in
  regular schemas, typically keyed by `(scenario_id, key)` for nodes and
  `(scenario_id, source_key, target_key)` for edges.

- **Extraction helpers** bridge these worlds:

  - They use `graph.vertices` to map indices ⇄ keys.
  - They use `graph.vertex_masks` to apply `GraphMask`‑based vertex filters.
  - They return either Pandas data frames or GraphBLAS vectors/matrices that
    are aligned with the `Graph`’s vertex indexing.

Design goals:

- Keep `Graph` minimal, pure python‑graphblas, and easy to test in isolation.
- Keep SQLAlchemy integration in a thin `db/extract` layer.
- Make it cheap and explicit to construct shallow graph views (different masks,
  same structure).
- Support incremental labelling without rebuilding the entire label matrix.


### 7.2 Relational Schema (`disco.graph.schema`)

All structural and mask‑related tables live in the `graph` schema. The SQLAlchemy
metadata is defined in `disco.graph.schema` and created via
`create_graph_schema(engine)`.

#### 7.2.1 Scenarios

```text
graph.scenarios
-------------------------------
scenario_id   TEXT PRIMARY KEY
created_at    TIMESTAMP NOT NULL
description   TEXT NULL
```

- `scenario_id` is the external identifier used throughout the system.
- `created_at` is set by `create_scenario`.
- `description` is optional metadata only.

#### 7.2.2 Vertices

```text
graph.vertices
-------------------------------------------------------------
scenario_id   TEXT   PK, FK → graph.scenarios.scenario_id
index         BIGINT PK     -- 0 .. V-1
key           TEXT   NOT NULL
```

- `(scenario_id, index)` identifies a vertex in *index space*.
- `key` is the *model key* for the vertex (e.g. node ID in model tables).
- All structural edges and masks refer to vertices via `(scenario_id, index)`.
- Model tables typically use `(scenario_id, key)`, so `vertices` is the bridge.

#### 7.2.3 Edges

```text
graph.edges
----------------------------------------------------------------------------------------------
scenario_id   TEXT   PK, FK → graph.scenarios.scenario_id
layer_idx     INT    PK     -- layer identifier (0 .. L-1)
source_idx    BIGINT PK     -- FK → graph.vertices(index)
target_idx    BIGINT PK     -- FK → graph.vertices(index)
weight        FLOAT  NOT NULL
```

- Each row represents a directed edge `source_idx → target_idx` in `layer_idx`.
- `(scenario_id, layer_idx, source_idx, target_idx)` must be unique.
- The in‑memory `Graph` assumes:
  - edges are always between valid vertex indices `0 .. num_vertices-1`,
  - layer indices are contiguous `0 .. num_layers-1`.

These invariants are validated when constructing and loading graphs.

#### 7.2.4 Labels and Vertex Labels

```text
graph.labels
--------------------------------------------------------
id           INT PRIMARY KEY
scenario_id  TEXT NOT NULL, FK → graph.scenarios.scenario_id
type         TEXT NOT NULL   -- label_type
value        TEXT NOT NULL   -- label_value

graph.vertex_labels
--------------------------------------------------------------------
scenario_id   TEXT   PK, FK → graph.scenarios.scenario_id
vertex_index  BIGINT PK, FK → graph.vertices.index
label_id      INT    PK, FK → graph.labels.id
```

- `graph.labels` defines label *kinds* for a scenario:
  - Each row has a `type` (e.g. `"location"`, `"echelon"`) and a `value`
    (e.g. `"EU"`, `"upstream"`).
  - `id` is a DB surrogate key and is re‑indexed into a dense label index
    for the in‑memory `Graph`.

- `graph.vertex_labels` maps vertices to labels:
  - Each `(scenario_id, vertex_index, label_id)` means “this vertex has
    this label”.

When loading a graph, DB label IDs are sorted by `labels.id` and mapped to dense
`0 .. num_labels-1` indices that are used as column indices in
`Graph.label_matrix`.

#### 7.2.5 Vertex Masks

```text
graph.vertex_masks
-----------------------------------------------------------------------------------------------
scenario_id   TEXT   PK, FK → graph.scenarios.scenario_id
mask_id       TEXT   PK     -- UUID string identifying a mask
vertex_index  BIGINT PK, FK → graph.vertices.index
updated_at    TIMESTAMP NOT NULL
```

- A *mask* is an arbitrary subset of vertices in a scenario, identified by a
  random `mask_id` (UUID).
- For each `(scenario_id, mask_id)` there are zero or more `vertex_index` rows.
- `updated_at` is touched whenever a mask is ensured/persisted.

`GraphMask` is responsible for populating and maintaining this table; it is used
from `graph.extract` and other higher‑level helpers.


### 7.3 Core Graph Representation (`disco.graph.core`)

`Graph` is the in‑memory representation of a layered directed graph backed by
python‑graphblas. It is completely DB‑agnostic.

```python
class Graph:
    def __init__(
        self,
        layers: tuple[Matrix, ...],
        num_vertices: int,
        scenario_id: str = "",
        *,
        mask: Vector | None = None,
        label_matrix: Matrix | None = None,
        label_meta: dict[int, tuple[str, str]] | None = None,
    ) -> None:
        ...
```

Key fields (via `__slots__`):

- `_layers: tuple[Matrix, ...]` — adjacency matrices per layer.
- `num_vertices: int` — number of vertices (0 .. num_vertices-1).
- `scenario_id: str` — scenario identifier (for context, not required).
- `_mask: GraphMask | None` — optional persisted vertex mask wrapper.
- `_label_matrix: Matrix[BOOL] | None` — sparse vertex×label matrix.
- `_label_meta: dict[int, (label_type, label_value)]`.
- `_labels_by_type: dict[label_type, dict[label_value, label_index]]`.
- `_label_indices_by_type: dict[label_type, np.ndarray[int64]]`.
- `_label_index_by_type_value: dict[(label_type, label_value), label_index]`.
- `_num_labels: int` — number of label columns.

#### 7.3.1 Construction and Validation

- `Graph.from_edges(edge_layers, *, num_vertices, scenario_id="")`

  - `edge_layers` is a mapping `layer_idx → (src, dst, weights)` where each
    array is converted to `np.int64` (for indices) and proper dtype for
    weights.
  - For each layer a `gb.Matrix.from_coo` is constructed with shape
    `num_vertices × num_vertices`.
  - Layer indices must be contiguous (0 .. L-1); otherwise a `ValueError`
    is raised.
  - The constructor then calls `validate(check_cycles=False)`.

- `Graph.validate(check_cycles: bool = True)`

  - For each layer:

    - Verifies `nrows == num_vertices` and `ncols == num_vertices`.
    - Calls `has_self_loop` to forbid self‑loops.
    - Optionally calls `matrix_has_cycle` to detect cycles in the layer
      when `check_cycles=True`.

#### 7.3.2 Structural Accessors

- `layers → tuple[Matrix, ...]` — read‑only tuple of adjacency matrices.
- `get_matrix(layer: int) → Matrix` — raw adjacency matrix for `layer`.
- `get_out_edges(layer: int, vertex_index: int) → Vector` — outgoing edges
  (row slice) as a `Vector`.
- `get_in_edges(layer: int, vertex_index: int) → Vector` — incoming edges
  (column slice) as a `Vector`.

These methods do **not** apply masks; masking is handled at higher levels by
`GraphMask` and the extraction helpers.

#### 7.3.3 Mask Handling (`GraphMask` integration)

Masks are represented in `Graph` as:

- `set_mask(mask_vector: Vector | None)`

  - If `mask_vector` is `None`, clears the mask.
  - Otherwise enforces:

    - `mask_vector.dtype == gb.dtypes.BOOL`.
    - `mask_vector.size == num_vertices`.

  - Wraps the vector in a `GraphMask(mask_vector, scenario_id=self.scenario_id)`
    and stores it in `_mask`.

- `mask_vector → Vector | None`

  - Returns the underlying GraphBLAS mask vector (or `None`).

- `graph_mask → GraphMask | None`

  - Returns the `GraphMask` wrapper (or `None`).

`GraphMask` is responsible for persisting its vector to `graph.vertex_masks`
via `ensure_persisted(session)`. This is used by `graph.extract` helpers when
joining filtered subsets of vertices or edges.

#### 7.3.4 Labels and Incremental Labelling

`Graph` stores vertex labels in a sparse matrix and multiple metadata
structures to support efficient per‑type queries.

- `set_labels(label_matrix: Matrix, label_meta: dict[int, (str, str)] | None)`

  - Attaches an existing label matrix and metadata.
  - Requires `label_matrix.dtype == BOOL` and
    `label_matrix.nrows == num_vertices`.
  - Sets `_num_labels = label_matrix.ncols`.
  - Rebuilds all per‑type structures (`_labels_by_type`, `_label_indices_by_type`,
    `_label_index_by_type_value`) from `_label_meta`.

- `label_matrix → Matrix | None` — sparse `num_vertices × num_labels` matrix.
- `label_meta → Mapping[int, (str, str)]` — read‑only view of label metadata.
- `num_labels → int` — number of labels (columns).

Per‑type helpers:

- `labels_for_type(label_type: str) → (np.ndarray, Matrix)`

  - Returns `(label_indices, submatrix)` where:

    - `label_indices` is a `np.ndarray[int64]` of global label indices of this type.
    - `submatrix` is a `Matrix[BOOL]` with all vertices and only those columns.

- `label_value_to_index(label_type: str) → Mapping[str, int]`

  - Read‑only mapping `label_value → label_index` for a type.

- `label_index_to_value_for_type(label_type: str) → Mapping[int, str]`

  - Inverse mapping `label_index → label_value` for a type.

- `label_info(label_index: int) → (label_type, label_value)`

  - Returns metadata for a label index, or raises if unknown.

- `get_vertices_for_label(label_id: int) → np.ndarray`

  - Returns indices of vertices that have label `label_id`.
  - Implemented via a column slice and a sparse `select("!=", 0)` operation.

Incremental labelling:

```python
def add_labels(
    self,
    label_type: str,
    labels: Mapping[str, np.ndarray | Sequence[int]],
) -> None:
    ...
```

- `labels` maps `label_value → sequence of vertex indices`.

- For each `(label_type, label_value)`:

  - If the `(type, value)` pair already exists, it reuses the existing
    label index.
  - Otherwise it allocates a new label index at `_num_labels` and updates
    `_label_meta`, `_labels_by_type`, `_label_index_by_type_value`, and
    the per‑type index set.

- Vertex indices are validated against `0 .. num_vertices-1`.

- A new boolean assignment is constructed with all existing non‑zero entries
  from `_label_matrix` plus all newly added `(vertex, label)` pairs and fed
  into `gb.Matrix.from_coo` with `dup_op=gb.binary.lor`:

  - This makes `add_labels` semantics a **set union**: if a `(vertex, label)`
    pair exists multiple times, the result is still `True` and no error is
    raised for duplicate indices.

- Per‑type index arrays (`_label_indices_by_type`) are rebuilt for affected
  label types at the end.

This allows label metadata and assignments to be built incrementally from
multiple calls without re‑allocating the entire matrix from scratch in user code.


#### 7.3.5 Distinct Label Combinations

The method:

```python
def by_distinct_labels(self, distinct: Sequence[str]) -> np.ndarray:
    ...
```

computes distinct combinations of labels across one or more label types.

- Input: a sequence of label types, e.g. `["location", "echelon"]`.
- For each type, `labels_for_type` gives:

  - `label_indices`: global label indices for that type.
  - `L_type`: boolean matrix `(num_vertices × num_labels_of_type)`.

- These type‑specific matrices are combined row‑wise using a Kronecker‑like
  operation implemented by `_rowwise_and_kron` and `_rowwise_and_kron_many`:

  - The result `S` has shape `(num_vertices × num_combinations)` and has `True`
    exactly at positions `(vertex, combination)` where the vertex has all labels
    involved in that combination.

- Column‑wise reduction `S.reduce_columnwise("lor")` identifies which label
  combinations are actually present in at least one vertex.

- The column indices of such combinations are decoded back into the original
  global label indices per type, and returned as a `np.ndarray` of shape
  `(num_combinations, len(distinct))`.

If each vertex carries at most one label per type, then each vertex contributes
to at most one combination, but the implementation is generic enough to handle
multiple labels per type as well.


#### 7.3.6 Graph Views

`Graph.get_view(mask: Vector | None = None) → Graph` constructs a shallow view:

- Structural matrices (`layers`) and label structures are shared.
- A new `Graph` instance is created with the same `num_vertices` and
  `scenario_id`.
- The mask used for the new graph:

  - If `mask is None`, it reuses the current mask vector (if any).
  - Otherwise it applies the given GraphBLAS `Vector[BOOL]`.

This is useful when computations need to work on different vertex subsets while
sharing the same underlying adjacency matrices and labels.


### 7.4 Scenario Creation, Storage, and Loading (`disco.graph.db`)

The `disco.graph.db` module deals with the relational representation of graphs.

#### 7.4.1 Scenario Creation

```python
def create_scenario(
    session: Session,
    scenario_id: str,
    vertex_keys: np.ndarray,
    *,
    description: str | None = None,
    replace: bool = False,
    chunk_size: int = 10_000,
) -> str:
    ...
```

Responsibilities:

- Ensure that a scenario with `scenario_id` does not already exist:

  - If it exists and `replace=False`, raise `ValueError`.
  - If it exists and `replace=True`, call `delete_scenario` to remove all
    related data before recreating.

- Insert a row into `graph.scenarios` with `created_at = utcnow()` and
  optional `description`.

- Insert vertices in chunks into `graph.vertices`:

  - `vertex_keys` is a 1‑D NumPy array (or array‑like) of keys.
  - Position `i` corresponds to vertex index `i` in the structural graph.
  - Vertices are inserted in batches of size `chunk_size` to avoid building
    very large insert lists in memory.

This function does **not** commit; the caller’s session management is
responsible for commit/rollback.

#### 7.4.2 Scenario Deletion

```python
def delete_scenario(session: Session, scenario_id: str) -> None:
    ...
```

Removes everything for a scenario in the correct dependency order:

1. `graph.vertex_masks`
2. `graph.vertex_labels`
3. `graph.edges`
4. `graph.labels`
5. `graph.vertices`
6. `graph.scenarios`

This again does not commit; transaction boundaries are the caller’s responsibility.

#### 7.4.3 Storing Graph Structure and Labels

```python
def store_graph(
    session: Session,
    graph: Graph,
    *,
    store_edges: bool = True,
    store_labels: bool = True,
) -> None:
    ...
```

- `_store_edges_for_scenario`:

  - Deletes existing rows in `graph.edges` for the scenario.
  - For each layer, calls `Matrix.to_coo()` and inserts rows with
    `(scenario_id, layer_idx, source_idx, target_idx, weight)`.

- `_store_labels_for_scenario`:

  - If `graph.label_matrix is None` or `graph.num_labels == 0`, does nothing.
  - Deletes existing rows in `graph.vertex_labels` and `graph.labels` for
    the scenario.
  - Inserts one row in `graph.labels` for each label index in `graph.label_meta`,
    returning the DB `id` for each inserted row.
  - Builds a mapping `label_index → labels.id`.
  - Iterates over non‑zero entries of `label_matrix` (`to_coo()`), and inserts
    rows into `graph.vertex_labels` with `(scenario_id, vertex_index, label_id)`
    for all entries that are `True`.

Again, no commit is performed here.

#### 7.4.4 Loading a Graph from the Database

```python
def load_graph_for_scenario(session: Session, scenario_id: str) -> Graph:
    ...
```

Steps:

1. `_load_num_vertices`:

   - Queries `max(vertices.index)` for the scenario and returns
     `max_index + 1` (or `0` if no vertices exist).

2. `_load_edge_layers`:

   - Queries `graph.edges` for the scenario.
   - Groups rows by `layer_idx`.
   - Creates a GraphBLAS matrix per layer via `Matrix.from_coo`.
   - Ensures layer indices are contiguous `0 .. num_layers-1`.

3. `_load_labels_for_scenario`:

   - Queries `graph.labels` for the scenario and sorts by `id`.
   - Assigns dense label indices `0 .. num_labels-1` and builds `label_meta`.
   - Queries `graph.vertex_labels` and uses `label_id_to_index` to build a
     sparse `label_matrix` via `Matrix.from_coo`. If there are no assignments,
     it constructs an empty boolean matrix with the appropriate size.

4. Constructs and returns a `Graph`:

   ```python
   Graph(
       layers=edge_layers,
       num_vertices=num_vertices,
       scenario_id=scenario_id,
       label_matrix=label_matrix,
       label_meta=label_meta,
   )
   ```

The resulting `Graph` is fully populated structurally and with labels, but
has no mask set by default.


### 7.5 Data Extraction and Maps (`disco.graph.extract`)

The `disco.graph.extract` module bridges structural graphs and model tables.
It uses the `graph.vertices` mapping and optional `GraphMask` instances to:

- Pull vertex‑aligned data into Pandas or GraphBLAS structures.
- Pull edge‑aligned data (per layer) into Pandas.
- Construct GraphBLAS adjacency‑like matrices filtered by masks.

Key concepts:

- **IndexBy**: whether results are indexed by vertex **index** or **key**.
- **EdgeIndexBy**: whether edge results are indexed by source/target **indices**
  or **keys**.
- **GraphMask** integration: when a mask is provided (or attached to the graph),
  only vertices or edges that belong to the mask are returned.

#### 7.5.1 Vertex Data via Index→Key Mapping

```python
def get_vertex_data(
    session: Session,
    graph: Graph,
    vertex_table: Table,
    columns: Sequence[ColumnElement[Any]],
    *,
    mask: GraphMask | None = None,
    index_by: Literal["index", "key"] = "index",
    default_fill: Any | None = None,
) -> pd.DataFrame:
    ...
```

- `vertex_table` is a *model node table* keyed by `(scenario_id, key)`.
- The function:

  - Starts from `graph.vertices` (`vertices_table`), which has
    `(scenario_id, index, key)`.
  - Optionally joins `graph.vertex_masks` using the provided or attached
    `GraphMask` to filter vertices.
  - LEFT OUTER JOINs the model node table on `(scenario_id, key)`.
  - Returns a DataFrame with the requested columns plus `index` and `key`.

- `index_by` controls the resulting index:

  - `"index"` → DataFrame indexed by vertex index.
  - `"key"` → DataFrame indexed by vertex key.

- `default_fill` is an optional value to `fillna` for missing model rows.

Numeric vector extraction:

```python
def get_vertex_numeric_vector(
    session: Session,
    graph: Graph,
    vertex_table: Table,
    value_column: ColumnElement[Any],
    *,
    mask: GraphMask | None = None,
    default_value: float = 0.0,
) -> gb.Vector:
    ...
```

- Similar join pattern but returns a `Vector[FP64]` of size `graph.num_vertices`.
- Missing values become `default_value`; if `default_value == 0.0` they are
  simply omitted in the sparse representation.

#### 7.5.2 Edge Data (Outbound / Inbound), Key‑based Tables

Model edge tables are expected to be key‑based:

```text
scenario_id   TEXT
source_key    TEXT
target_key    TEXT
...
```

Validation helper:

```python
def _validate_edge_table(edge_table: Table) -> None:
    # requires columns: scenario_id, source_key, target_key
```

Outbound edges:

```python
def get_outbound_edge_data(
    session: Session,
    graph: Graph,
    edge_table: Table,
    columns: Sequence[ColumnElement[Any]],
    *,
    layer_idx: int,
    mask: GraphMask | None = None,
    index_by: Literal["indices", "keys"] = "indices",
    default_fill: Any | None = None,
) -> pd.DataFrame:
    ...
```

- Starts from `graph.edges` for the given `layer_idx`.
- Joins `graph.vertices` twice (source and target) to get `source_key` and
  `target_key`.
- LEFT OUTER JOINs the model edge table on `(scenario_id, source_key, target_key)`.
- Optionally filters by `GraphMask` on the *source* index via `graph.vertex_masks`.
- Returns a DataFrame indexed by:

  - `("source_index", "target_index")` if `index_by == "indices"`, or
  - `("source_key", "target_key")` if `index_by == "keys"`.

Inbound edges:

```python
def get_inbound_edge_data(
    session: Session,
    graph: Graph,
    edge_table: Table,
    columns: Sequence[ColumnElement[Any]],
    *,
    layer_idx: int,
    mask: GraphMask | None = None,
    index_by: Literal["indices", "keys"] = "indices",
    default_fill: Any | None = None,
) -> pd.DataFrame:
    ...
```

- Same pattern as outbound, but mask (if any) is applied on the *target* index.

#### 7.5.3 Map Extraction (GraphBLAS Matrices from `graph.edges`)

Two helpers provide sparse adjacency matrices derived directly from
`graph.edges`:

```python
def get_outbound_map(
    session: Session,
    graph: Graph,
    *,
    layer_idx: int,
    mask: GraphMask | None = None,
    value_source: Literal["weight"] = "weight",
) -> gb.Matrix:
    ...
```

- Queries `graph.edges` for the scenario and `layer_idx`.
- If a `GraphMask` is provided or attached to the graph, joins
  `graph.vertex_masks` and filters by the *source* vertex.
- Returns a `Matrix[FP64]` with shape `num_vertices × num_vertices` populated
  with edge weights for the remaining edges.

Inbound map:

```python
def get_inbound_map(
    session: Session,
    graph: Graph,
    *,
    layer_idx: int,
    mask: GraphMask | None = None,
    value_source: Literal["weight"] = "weight",
) -> gb.Matrix:
    ...
```

- Same semantics but filters by the *target* vertex in the mask.

Currently, `value_source` only supports `"weight"`; this is kept as an enum to
allow future extensions (e.g. taking values from model edge tables).


### 7.6 Summary

The graph subsystem cleanly separates concerns:

- `Graph` is a DB‑agnostic, python‑graphblas‑based representation of layered
  directed graphs with labels and masks.
- `graph.schema` defines the relational schema for scenarios, vertices, edges,
  labels, vertex labels, and vertex masks.
- `graph.db` handles scenario lifecycle and round‑trips between DB structure
  and in‑memory graphs.
- `graph.extract` provides high‑level helpers to merge structural graphs with
  model data and masks into Pandas and GraphBLAS structures.

This separation allows simulation and optimization code to operate purely on
`Graph` and GraphBLAS structures, while still supporting robust storage,
querying, and integration with relational model data.


## 8. Model

This chapter specifies the **disco.model** subpackage: how simulation *models* (node implementations + model metadata)
are packaged, discovered, validated, and loaded at runtime.

A **Model** defines **types** (simprocs and node types). It does **not** define the scenario instance.
The **Scenario** lives in the database and defines the **Layered DAG instance** (vertices and edges) and stores
scenario-specific node/edge data rows.

In addition, the Model subsystem attaches a **SQLAlchemy table model** (“ORM bundle”) so that core components
(Graph, scenario readers, etc.) can access node and edge data via SQLAlchemy `Table` objects in a consistent way.

### 8.1 Purpose and scope

The Model subpackage provides:

- A **validated schema** for `model.yml` using **Pydantic** (v2).
- A **loader** that discovers and loads models as **Python packages** (recommended), while also supporting a
  local “dev folder” mode.
- A runtime **Model** object that exposes:
  - `spec: ModelSpec` (the parsed+validated `model.yml`)
  - `node_classes: dict[str, type]` (imported Node implementation classes)
  - `orm: OrmBundle` (SQLAlchemy `MetaData` + resolved node/edge tables)
  - `node_factory(node_type)` to construct a node object (not yet initialized with runtime resources)

The Model system is designed for distributed runs where the **Orchestrator** and **Workers** may run in separate
processes and on separate machines. For this reason, each **Worker loads the Model locally from disk** (via the
installed model package or a local folder in development). Models are not transmitted as code over the network.

### 8.2 Conceptual separation

The system distinguishes four concepts:

- **Model**
  - defines simprocs and node types
  - contains the Node implementation code (Python) for node types
  - defines/attaches SQLAlchemy table models for scenario node/edge data tables (see 8.4.4)
- **Scenario (database)**
  - defines the Layered DAG instance (vertices and edges)
  - stores node/edge data rows in scenario tables referenced by the Model
- **Experiment**
  - defines parameters passed to node initialization
  - defines replications (and their seeds)
- **PartitioningScheme**
  - defines how vertices are allocated to node instances
  - defines how node instances are allocated to partitions

The **Partitioner** belongs to the Orchestrator runtime (not to Workers). Workers need the Model to:
instantiate nodes and interpret scenario data (via Graph/database code).

### 8.3 Model package requirements

A Model is distributed as a **separate Python package** (wheel / editable install). The `disco` distributions
must **not** contain any model implementations.

A model package should:

- declare a dependency on `disco` and its own dependencies (e.g., `numpy`, `numba`, `sqlalchemy`)
- ship a `model.yml` file as **package data**
- optionally ship an ORM module that defines SQLAlchemy `MetaData`/`Table` objects (see 8.4.4)
- provide an entry point in group `disco.models` (see 8.6.1)

Example package layout:

```ticks
my_model_pkg/
  pyproject.toml
  src/my_model_pkg/
    __init__.py
    model.yml
    nodes/
      __init__.py
      warehouse.py
      factory.py
    orm.py
    kernels/
      __init__.py
      inventory_kernels.py
```

### 8.4 model.yml schema

The model specification is stored in `model.yml` and validated by `ModelSpec` (Pydantic).

#### 8.4.1 simprocs

Simprocs define the **ordered process levels** in the simulation. Simproc order is implicit and defined by the
ordering of entries in `model.yml`.

Two forms are supported:

1. **Simple form (list of names)**  
   Use when edge data is stored only in the default edge table.

   ```ticks
   simprocs:
     - demand
     - supply
   ```

2. **Rich form (mapping)**  
   Use when one or more simprocs have a dedicated edge data table. The order is the YAML mapping insertion order.

   ```ticks
   simprocs:
     demand:
       edge-data-table: demand_edges
     supply: {}
   ```

Validation rules:

- simprocs must not be empty
- simproc names must be **non-empty** and **unique**
- in mapping form, `edge-data-table` (if present) must be a non-empty string
- if multiple simprocs specify `edge-data-table`, table names must be unique (to avoid ambiguous routing)

#### 8.4.2 node-types

`node-types` is a mapping from **node type name** to a node type specification:

- `class`: import reference to the Node implementation class (`pkg.module:ClassName` or `pkg.module.ClassName`)
- `node-data-table`: scenario table name containing node-instance data for this node type
- `distinct-nodes`: optional list of vertex label attributes used by partitioning to split vertices into distinct
  node instances
- `self-relations`: optional list of `(higher_simproc_name, lower_simproc_name)` tuples representing relations from
  a higher-level simproc to a lower-level simproc *on the same node instance*; ordering constraint:
  `order(higher) < order(lower)`

Example:

```ticks
node-types:
  Warehouse:
    class: my_model_pkg.nodes.warehouse:Warehouse
    node-data-table: warehouse_data
    distinct-nodes: [site]
    self-relations:
      - [demand, supply]
```

Validation rules:

- node type keys must be non-empty
- `class` must parse as a module + class reference
- `self-relations` must reference known simproc names and must point from higher to lower (smaller index to larger index)

#### 8.4.3 Edge data tables

Edges have no logic/implementation. The Model only maps **where edge data lives** in the Scenario database.

Two mechanisms exist:

- `default-edge-data-table` (optional): scenario table for edges that do not use simproc-specific edge data tables.
- simproc-specific `edge-data-table` (optional, mapping form only): scenario table for edges at that simproc level.

Example with both:

```ticks
default-edge-data-table: edges_default

simprocs:
  demand:
    edge-data-table: demand_edges
  supply: {}
```

Interpretation:

- For simprocs that define an `edge-data-table`, edge data for that simproc is expected in that table.
- For simprocs without `edge-data-table`, edge data is expected in `default-edge-data-table` (if configured).
- A model may omit simproc-specific edge tables entirely and only use the default edge table.

#### 8.4.4 ORM configuration

The Model loader attaches SQLAlchemy `Table` objects for all node/edge data tables referenced in `model.yml`.

Two modes are supported:

1. **Python ORM provider (preferred)**  
   `model.yml` contains an `orm` reference to a callable that returns `sqlalchemy.MetaData`.  
   The `MetaData` must contain the tables referenced by `node-data-table`, `default-edge-data-table`, and any
   per-simproc `edge-data-table` entries.

   Example:

   ```ticks
   orm: my_model_pkg.orm:get_metadata
   ```

   With:

   ```ticks
   # my_model_pkg/orm.py
   from sqlalchemy import MetaData
   metadata = MetaData()
   def get_metadata() -> MetaData:
       return metadata
   ```

   If `AppSettings.database.create_tables` is `True`, the loader may call `metadata.create_all(engine)`
   before validation, allowing first-time setup in development/test environments.

2. **Reflection mode**  
   If `orm` is not provided, the loader reflects the required tables from the database using the table names
   from `model.yml`. In this mode, all referenced tables **must already exist** in the database.

### 8.5 Runtime Model object

The loader produces a runtime `Model`:

- `Model.spec`: the validated `ModelSpec` (all model.yml metadata)
- `Model.node_classes`: mapping from node type name to imported Python class
- `Model.orm`: an `OrmBundle` containing:
  - `metadata: sqlalchemy.MetaData`
  - `node_tables: dict[str, sqlalchemy.Table]` keyed by node type name
  - `default_edge_table: Optional[sqlalchemy.Table]` for the default edge table (if configured)
  - `edge_tables_by_simproc: dict[str, sqlalchemy.Table]` mapping simproc name -> edge table (only for simprocs that define `edge-data-table`)

The Model is intentionally lightweight. Convenience accessors should generally be added as pure helper functions
instead of inflating the `Model` interface; consumers can access metadata via `model.spec` and tables via
`model.orm`.

### 8.6 Model discovery and loading

Models are discovered and loaded in one of three ways:

1. **Entry point plugin name** (recommended for production)
2. **Direct package name**
3. **Local dev folder** (fallback; recommended dev approach is still editable install)

A unified facade exists (signatures simplified):

```ticks
load_model(plugin=..., conn_or_engine=..., create_tables=...)
load_model(package=..., conn_or_engine=..., create_tables=...)
load_model(path=..., dev_import_root=..., conn_or_engine=..., create_tables=...)
```

Exactly one of `plugin`, `package`, `path` must be provided.

Because ORM attachment may require reflection and/or table creation, the loader requires an SQLAlchemy
`Engine` or `Connection` (`conn_or_engine`) and a `create_tables` flag (typically sourced from
`AppSettings.database.create_tables`).

#### 8.6.1 Entry points (strict contract)

Model packages register an entry point in group `disco.models`. The entry point is a callable that returns a
**package name string**. This strict contract keeps loading logic centralized and deterministic.

Example `pyproject.toml`:

```ticks
[project.entry-points."disco.models"]
supply_chain_v1 = "my_model_pkg.plugin:model_package"
```

Example `my_model_pkg/plugin.py`:

```ticks
def model_package() -> str:
    return "my_model_pkg"
```

The loader then performs:

- import the returned package
- read `model.yml` from the package resources
- validate using `ModelSpec`
- import Node classes referenced in `node-types.*.class`
- attach ORM tables (Python ORM provider or reflection) and validate table contracts (see 8.7)

If an entry point returns anything other than a non-empty string, model loading fails.

#### 8.6.2 Package loading

When loading by package name, `model.yml` is read using `importlib.resources` from within the package. This
provides stable module names and is well-suited for distributed deployments and Numba caching.

The model package must include `model.yml` as package data.

#### 8.6.3 Local dev folder mode

A fallback loader is available for local development without installing the model package. In this mode:

- `model.yml` is read from `<path>/model.yml`
- the loader temporarily prepends `dev_import_root` (or `path`) to `sys.path` to import modules referenced in `class`
  and (if present) the ORM provider module referenced by `orm`

This mode is best-effort and intended for development only. The preferred dev workflow is:

```ticks
pip install -e path/to/model
```

and then load via entry point or package name.

### 8.7 ORM table contracts and validation

After obtaining tables (provider or reflection), the loader validates a minimal schema contract.

#### 8.7.1 Node data tables

For each node type, the node data table must satisfy:

- Exactly one table is resolved (the table named by `node-data-table`).
- Mandatory columns:
  - `scenario_id` (string-like, **NOT NULL**)
  - `key` (string-like, **NOT NULL**)
- Uniqueness per scenario:
  - the pair (`scenario_id`, `key`) must be enforced as **exactly** the primary key
    *or* an exact **UNIQUE** constraint/index on (`scenario_id`, `key`).

#### 8.7.2 Edge data tables

Edge tables can be:

- the default edge table (`default-edge-data-table`), optional
- zero or more simproc edge tables (`simprocs.<name>.edge-data-table`), optional

Each edge table must satisfy:

- Mandatory columns:
  - `scenario_id` (string-like, **NOT NULL**)
  - `source_key` (string-like, **NOT NULL**)
  - `target_key` (string-like, **NOT NULL**)

### 8.8 Node construction and initialization lifecycle

The Node implementation interface is provided by the core `disco.node.Node` abstract base class.

Node lifecycle requirements:

- Node classes must be importable from the model package and must subclass `Node`.
- Node objects must be constructible via a lightweight, no-argument `__init__()`.
- Runtime resources and run-specific parameters are bound via `initialize(...)`, called by the NodeController.

The Model must not initialize nodes with experiment/replication resources. Instead:

1. NodeController calls `node = model.node_factory(node_type)` to construct the node object.
2. NodeController calls `node.initialize(experiment=..., replication=..., resources=..., params=...)`.

This design avoids “half-constructed objects” and keeps initialization control in the NodeController.

### 8.9 Numba compatibility and best practices

To maximize Numba performance and cache reuse:

- Prefer packaging models as real Python packages (stable module names).
- Place Numba kernels in dedicated modules (e.g., `my_model_pkg.kernels.*`) and compile with `cache=True`.
- Keep kernels operating on NumPy arrays / scalars; keep Node methods as orchestration code.
- Configure `NUMBA_CACHE_DIR` in the worker runtime if persistent caches are desired.

Dev folder imports that alter module naming or import paths can reduce cache hit rates; editable installs are the
recommended development approach.

### 8.10 Failure modes and diagnostics

The loader fails fast with clear errors for:

- missing or unreadable `model.yml`
- invalid YAML or schema violations in `model.yml`
- missing modules / missing classes referenced by `class`
- ORM provider misconfiguration (cannot import provider, provider not callable, provider returns non-`MetaData`)
- missing tables in provider metadata (Python ORM mode) or missing tables in the database (reflection mode)
- table schema contract violations (missing required columns, nullable required columns, missing uniqueness constraints)
- Node classes that do not subclass `disco.node.Node`
- entry point misconfiguration (no plugin found, non-callable target, non-string return value)

These failures should be treated as deployment/configuration errors and must be resolved before starting a run.


## 9. Event Queue

This chapter specifies the **EventQueue** abstraction used by nodes/transports to deterministically process
incoming events from multiple predecessors in **epoch order**, while supporting **out-of-order arrival**
of events and **late knowledge** of per-epoch event counts.

There is exactly one **EventQueue** per **NodeRuntime / SimProc**. All semantics in this chapter apply to that unit.

The implementation below reflects the reference (pure Python) `EventQueue` logic in `event_queue.py` and the
expected semantics used by the test-suite.

### 9.1 Purpose and core idea

For a given NodeRuntime and SimProc, the runtime receives events from multiple upstream (node, simproc) predecessors. 
To process events safely and deterministically, the node must only **enable** an epoch **E** when:

1. Every predecessor has **defined** epoch **E** (via promises), i.e. the queue knows that **E** is a real epoch
   boundary (and not merely “unknown future”), and
2. For epoch **E**, the promised number of events has been **received** from every predecessor that participates in **E**, and
3. No **earlier** epoch still contains unpopped events.

The **EventQueue** maintains this discipline and exposes:

- `epoch`: the currently enabled epoch (events may be popped for this epoch).
- `next_epoch`: the earliest **defined** future epoch across all predecessors (or `None` if unknown).
- `waiting_for`: a human-readable reason why progress is blocked.

### 9.2 Data model

The system has two layers:

#### 9.2.1 EventQueue (simproc-level aggregator)

`EventQueue` manages:

- A set of named predecessor queues: `dict[str, PredecessorEventQueue]`.
- Global state:
  - `_epoch: float` (enabled epoch; `-1.0` means not initiated; `inf` if no predecessors)
  - `_next_epoch: float | None` (earliest defined next epoch; `None` if not yet known)
  - `_waiting_for: str | None`

#### 9.2.2 PredecessorEventQueue (per-predecessor state)

Each predecessor queue maintains:

- `_promises: list[Promise]` where `Promise = (epoch: float, seqnr: int, num_events: int)`
  - `seqnr` is strictly increasing for *new* promises.
- `_num_events: dict[int, int]` mapping `seqnr -> promised_count`
- `_num_events_received: dict[int, int]` mapping `seqnr -> received_count`
- `_events: heap[(epoch, counter, data, headers)]` to store events and preserve insertion order for ties.
- `_seqnr: int` the **last completed** promise seqnr (initially `0`)
- Derived properties:
  - `epoch`: epoch of `_seqnr` (i.e., last completed epoch; `-1.0` if none completed yet)
  - `next_epoch`: epoch of `_seqnr + 1` if defined, else `None`
  - `empty`: whether there are events available **for the predecessor's current epoch**
  - `waiting_for_promise` / `waiting_for_events`

**Important:** For a predecessor, the “current epoch” is the epoch of the **last completed promise**
(i.e., `_seqnr`). Events in the heap for that epoch are ready to be popped.

### 9.3 Epoch state definitions

Throughout this chapter we use the following terminology (these apply at node-level and predecessor-level,
depending on context):

- **defined** (epoch time is known): a promise exists such that the epoch time is known to be an epoch boundary.
- **complete** (all events received): for that epoch, the promised number of events has been received.
- **enabled** (safe to pop): the epoch is complete **and** there is no earlier epoch anywhere in the queue
  with pending (unpopped) events.

Notes:
- An epoch can be **complete** but not **enabled** if an earlier epoch still has unpopped events.
- An enabled epoch is necessarily complete.
- If there are no predecessors, `epoch = inf` and `next_epoch = inf`.

### 8.4 Promises

A **promise** tells the receiver how many events to expect for `(seqnr, epoch)` from a specific predecessor.

#### 9.4.1 Promise constraints (per predecessor)

For a predecessor `P`, promises must follow:

1. **Monotonic sequence numbers**
   - A *new* promise must have `seqnr == len(_promises) + 1`.
2. **Non-decreasing epochs**
   - For `i < j`, `epoch[i] <= epoch[j]`.
   - If a new promise arrives with an earlier epoch than the last promised epoch: **RuntimeError**.
3. **Promise extension for the same epoch (new seqnr)**
   - If the new promise is for the *same* epoch as the last promise and uses a new `seqnr`,
     the counts are **added**. This supports producing one epoch in multiple batches.
4. **Renewed promise (same seqnr)**
   - A promise with `seqnr == len(_promises)` is a **renewal**.
   - Renewals are used for **pre-announcing** an epoch boundary with an upper bound and later refining it.
   - Renewal rules:
     - `epoch` must match the last promise epoch, else **RuntimeError**.
     - The renewed `num_events` must be **strictly lower** than the currently promised count.
       If the renewed `num_events` is **greater than or equal** to the current promised count,
       the renewal is treated as **stale/out-of-order** (an earlier message that arrived late) and is **ignored**.
     - Lowering the count below already received events is **RuntimeError**.

This supports the common pattern: promise a conservative upper bound (e.g. a very large value) to make the epoch
boundary **defined**, and later lower it once the true value is known. It also tolerates late delivery of older
renewals: if a renewal would increase (or not decrease) the promised count, it is assumed to be stale and ignored
rather than failing.

#### 9.4.2 What “completing a promise” means

For a given predecessor and promise seqnr **k**:

- The promise is **complete** when `received_count[k] == promised_count[k]`.
- When promise **k** completes, the predecessor advances `_seqnr` to **k** and can expose the events for that
  promise's epoch for popping.

### 9.5 Events

Events arrive as `(sender, epoch, data, headers)`.

#### 9.5.1 Constraints

For a predecessor queue:

- An event must be strictly in the **future** relative to the predecessor’s current epoch:
  `event.epoch > predecessor.epoch`.

This prevents delivering additional events for an already-completed epoch.

Events are stored in a heap keyed by `(epoch, insertion_order)` so:

- Epoch ordering is deterministic.
- Events within the same epoch preserve arrival order.

#### 9.5.2 Receiving an event increments received counts

When an event for epoch **E** arrives, the predecessor identifies which promise seqnr **k** corresponds
to epoch **E** and increments `received_count[k]`. This may complete that promise and advance the
predecessor’s current epoch.

### 9.6 Enabling epochs at the node level

`EventQueue` enables epochs based on the state of **all** predecessors.

#### 9.6.1 `epoch` and `next_epoch` (canonical semantics)

- **epoch** (`queue.epoch`) is the time of the **enabled** epoch:
  an epoch for which a promise was made, all events have been received, and **no earlier events remain**
  to be popped.

- **next_epoch** (`queue.next_epoch`) is the time of the next **defined** epoch that is not yet enabled.
  If no next epoch is defined yet, `next_epoch` is `None`.

The node advances `epoch := next_epoch` if and only if:

1. There are no more events to pop for the current `epoch` (i.e., earlier epochs do not have pending events), and
2. `next_epoch` is defined and **complete** (all events for it have been received).

#### 9.6.2 Completeness vs enabling

- The **next** epoch can be **complete** without being **enabled** if there are still events pending in earlier epochs.
- An epoch cannot be enabled if there exists any earlier epoch with pending events (including across predecessors).

This means an “earlier complete-but-not-enabled” epoch cannot exist before an enabled epoch.
(See diagram rules in §9.7.)

#### 9.6.3 How `try_next_epoch()` chooses progress

At a high level, `try_next_epoch()`:

1. Maintains `epoch` as the earliest enabled epoch (the earliest epoch whose events are safe to pop).
2. Computes `next_epoch` as the earliest **defined** epoch strictly after `epoch`, if such a boundary is currently known.
   If any predecessor lacks a defined next epoch boundary, `next_epoch` becomes `None` and `waiting_for` indicates which.

### 9.7 ASCII epoch diagrams (notation)

The tests often use compact diagrams to describe promise/event states per predecessor and the merged result.

#### 9.7.1 Symbols

Symbols describe the state of *epochs* (not individual events):

- `*` = epoch is **enabled** (this is the current `queue.epoch` at the node level; enabled implies complete).
- `o` = epoch is **complete** but **not enabled** (blocked by earlier epochs with pending events).
- `-` = epoch is **defined** (promised) but **not complete** yet (still waiting for events).
- `.` = epoch (or further future) is **unknown** (no promise boundary is known yet).

#### 9.7.2 Validity rule (important)

`o----*` is **not valid** when `o` is earlier in time than `*`.

Reason: `*` (enabled) implies there is no earlier epoch anywhere that still has pending events. If an earlier
epoch were complete-but-not-enabled (`o`), that would mean some still-earlier epoch has pending events, which would
prevent `*` from being enabled.

Valid patterns include:
- `*--` (enabled epoch, next epochs defined but not complete)
- `*--o` (enabled now; a later epoch is complete but blocked)
- `*...o` (enabled now; immediate next boundary unknown; a later boundary is known and complete)

### 9.8 Public API and return-value semantics

#### 9.8.1 `register_predecessor(name: str)`

Registers a predecessor. Must be called before the queue is initiated (while `epoch == -1.0`).

#### 9.8.2 `promise(sender, seqnr, epoch, num_events) -> bool`

Routes the promise to the sender’s predecessor queue and recomputes `next_epoch` (and possibly enabling conditions).

Return value:
- `True` **iff** this call **updates `next_epoch`** (i.e., changes the value of `queue.next_epoch`).
- `False` otherwise (including the common case where a future promise is recorded but the queue is still
  waiting for events in earlier epochs, or where a **stale renewal** is ignored).

This aligns with tests that expect a future promise to return `False` when the current enabled epoch is still blocked.

#### 9.8.3 `push(sender, epoch, data, headers=None) -> bool`

Pushes an event to the sender’s predecessor queue; this may complete a promise and enable progress.

Return value:
- `True` **iff** this call **updates `epoch`** (i.e., changes the value of `queue.epoch` by enabling a new epoch).
- `False` otherwise (e.g., still waiting for events from another predecessor).

#### 9.8.4 `pop() -> iterator[(sender, epoch, data, headers)]`

Yields all events for the node’s current `epoch` across all predecessors, then attempts to enable subsequent epochs.

#### 9.8.5 Properties

- `epoch: float`
  - Enabled epoch; `inf` if there are no predecessors; `-1.0` before initiation.
- `next_epoch: float | None`
  - Earliest defined future epoch; `None` if unknown due to missing promises; `inf` if no predecessors.
- `empty: bool`
  - `True` if there are no further events for the current `epoch` across all predecessors.
- `waiting_for: str`
  - Explanation for why progress is blocked (missing initial promises, missing promises for next epoch, or missing events).

### 9.9 Error handling

The reference implementation intentionally treats certain protocol violations as programmer errors:

- Promises that go “back in time” (epoch decreases) raise **RuntimeError**.
- Renewed promises:
  - mismatching epoch raises **RuntimeError**,
  - lowering below already received events raises **RuntimeError**,
  - renewals that would **increase** (or not strictly decrease) the promised count are treated as **stale** and **ignored**
    (they do **not** raise).
- Events for epochs not strictly in the future of the predecessor’s current epoch are rejected (assertion / error).

These behaviors are designed to fail fast on true protocol bugs while remaining tolerant to **out-of-order delivery**
of older renewal messages.

### 9.10 Testing guidance

Tests should focus on these invariants:

- **Monotonicity:** seqnr increases for new promises; epochs do not decrease per predecessor.
- **Renewal semantics:**
  - valid renewals strictly lower counts and may unblock progress if they cause an epoch to become complete;
  - stale renewals (>= current promised count) are ignored.
- **Return values:** `promise()` returns `True` only when `next_epoch` changes; `push()` returns `True` only when `epoch` changes.
- **Multi-predecessor ordering:** `epoch` is always the earliest enabled epoch; later complete epochs may exist but cannot be enabled
  while earlier epochs still have pending events.


## 10. SimProc (Simulation Process)

This chapter specifies the **SimProc** abstraction used by Disco to model *one simulation process (timeline)* for *one node*.
A SimProc is the execution unit that advances simulation time (**epochs**) for a given *(node, simproc)* stream, based on:

- **incoming events** (from predecessor *(node, simproc)* pairs),
- **incoming promises** (from predecessor *(node, simproc)* pairs),
- **local wakeups** (self-scheduled future epochs), and
- **topology constraints** (allowed successors only).

A SimProc owns exactly one **EventQueue** and is advanced by the **NodeRuntime** attention loop.

### 10.1 Concept and relationship to the Graph

A SimProc can be seen as an **information layer** across the simulation network in which information flows along directed
edges (e.g. an **order** process flowing upstream and a **supply** process flowing downstream).

Key relationships:

- The **SimProc artifact is per node**: a `SimProc` instance represents the simulation process for a **single node**.
- Simulation processes are defined by the **Model** (e.g. `model.yml: simprocs`).
- The simulation **Graph** is layered: each layer corresponds to one simulation process.
- Per node, the number of SimProcs equals the number of Graph layers; **there is one SimProc per layer**.
- Within a single SimProc (i.e. within one layer), the predecessor/successor relationships between nodes form a **DAG**
  (directed acyclic graph).

A SimProc communicates with other SimProcs only through declared **predecessor** and **successor** links. These links are
addressed by *(node_name, simproc_name)* pairs.

### 10.2 Ownership and visibility

A SimProc is not addressed directly by model code:

- The **Node** (model logic) calls **NodeRuntime runtime methods** (e.g. “send event”, “set wakeup”).
- The **NodeRuntime forwards** these calls to the **active SimProc** (the SimProc currently executing the handler for the node).
- The transport/routing layer does not address a SimProc directly either; it routes envelopes to a node and the node’s
  runtime dispatches them to the correct SimProc based on `target_simproc`.

A SimProc owns:

- one `EventQueue` (exactly one per SimProc),
- wakeup state (soft/hard wakeups),
- successor communication state (promised epochs, sequence numbers, advance promises),
- an outbox for events to ensure promises are sent *before* events.

### 10.3 Epochs and advancement model

#### 10.3.1 Definitions

- **epoch**: the current simulation time of this SimProc for which all earlier enabled work has been processed.
- **next_epoch**: the earliest future epoch at which this SimProc can advance next. If unknown, `next_epoch = None`.

A SimProc advances by attempting to move from its current `epoch` to `next_epoch`. Advancement is triggered by the
NodeRuntime giving the SimProc attention in the runner loop.

#### 10.3.2 Initialization and epoch 0 guarantee

A SimProc starts with:

- `epoch = -1`,
- an implicit **hard wakeup at t = 0**, and therefore
- `next_epoch = 0` initially.

This is deliberate and provides a strong invariant:

> **For every SimProc of every node, the node handler is invoked for epoch 0** (per SimProc), even if the SimProc has
> no predecessors and even if the first incoming events from predecessors occur at a later time.

This invariant is essential for model initialization that is epoch-aware and for consistent start-of-simulation semantics.

#### 10.3.3 Advancement step (try_next_epoch)

On attention, a SimProc attempts to advance using the following high-level algorithm:

1. If `next_epoch` is `None`, the SimProc cannot advance (the next enabled epoch is unknown).
2. Drain any events that are available in the EventQueue **up to the current `next_epoch`**.
   - At startup, the queue epoch may still be `-1` even if epoch-0 events have arrived; draining therefore uses
     `queue.epoch <= next_epoch` to ensure epoch-0 events can be consumed prior to the epoch-0 handler call.
3. Decide whether the SimProc is allowed to advance to `next_epoch` based on:
   - no predecessors (pure wakeup-driven progress), or
   - the EventQueue indicating the next enabled epoch has been reached, or
   - a wakeup boundary being reached before the queue’s next enabled epoch.
4. If advancement is allowed:
   - set `epoch = next_epoch`,
   - flush wakeups that are now in the past,
   - invoke the node handler (via NodeRuntime) if there is a wakeup or any drained events,
   - recompute `next_epoch`,
   - make promises to successors,
   - then send events from the outbox (promises-first discipline).

The exact conditions and the interaction with `EventQueue.epoch` and `EventQueue.next_epoch` are normative and align with
the reference implementation.

### 10.4 Handler invocation and runtime façade

The SimProc’s handler is exposed as a callback (conceptually the node’s *event handler*). In code, this appears as
`on_events(simproc_name, events)`.

Handler invocation semantics:

- The handler runs **in the context of the active SimProc**.
- While the handler runs, the Node may:
  - send events to successors,
  - set wakeups,
  - make advance promises to specific successors,
  by calling NodeRuntime runtime methods, which forward to the active SimProc.

Important: the handler is **not permitted to emit promises directly**. Promises are produced automatically by the SimProc
based on outbox state and its `next_epoch` computation (see §10.5).

### 10.5 Promises

Promises are the mechanism that allows deterministic and concurrent execution with out-of-order message delivery.

A **promise** to a successor *(node, simproc)* states:

- an **epoch boundary** (time),
- a **sequence number** (`seqnr`), and
- a promised **maximum number of events** that will be sent to that successor for that epoch.

#### 10.5.1 Promise intent

Promises serve three purposes:

1. **Epoch existence**: they define which epochs are real boundaries downstream (no “unknown future” ambiguity).
2. **Event count completeness**: they allow a successor to enable an epoch only after it has received the promised number
   of events for that epoch from each predecessor participating in that epoch.
3. **Concurrency**: they allow upstream nodes to signal “nothing will happen until at least time T”, enabling downstream
   nodes to safely progress to that time without waiting for large messages to arrive.

#### 10.5.2 Prepromise and repromise

The SimProc promises in two phases per successor:

1. **Finalize earlier epochs** (where the event count is now known):
   - For all epochs `< next_epoch`, the SimProc sends a promise with the *known* event count for that epoch.
2. **Prepromise next_epoch** (where the event count is not yet known):
   - For `epoch = next_epoch`, the SimProc sends a promise with `num_events = MAX_UINT32` to mean “unknown / unbounded
     so far”, establishing the epoch boundary early.

Later, when it becomes known that fewer events will be sent for an epoch previously prepormised, the SimProc may
**repromise downward** for that epoch (i.e. send a promise for the same epoch with a smaller `num_events`).
This creates additional opportunities for concurrency downstream.

#### 10.5.3 Out-of-order delivery and “ignored” updates

Due to transports, promise messages may arrive out of order. Receivers (EventQueue) must therefore treat promises as
monotone refinements, using `(sender, seqnr, epoch)` and event-count rules to ignore stale information.

In particular, a receiver may ignore a promise that would increase the already-established number of events for an epoch
if the receiver concludes it was overtaken by a later promise (e.g. sequence-number based ordering, or other receiver-side
rules).

> The “ignore higher repromise” behavior is a receiver-side robustness feature against out-of-order message delivery.

### 10.6 Wakeups

A **wakeup** is an internal scheduling mechanism that inserts future epochs that are not derived from predecessor promises.
Wakeups are conceptually “events to self” and are used to model time-based behavior or to create concurrency windows.

There are two kinds:

- **soft wakeup**:
  - inserts an epoch into the future epoch sequence,
  - may be superseded by earlier promised epochs, and
  - is processed when it becomes the earliest enabled next epoch.
- **hard wakeup**:
  - declares that there will be **no future epochs before it** (except earlier hard wakeups),
  - causes all events and promises prior to the hard wakeup to be postponed,
  - is particularly useful to signal to successors that they may safely progress until that time, improving concurrency.

Every SimProc has an implicit **hard wakeup at epoch 0** (see §10.3.2).

### 10.7 Advance promises

An **advance promise** is a per-successor constraint that states:

> “This SimProc will not send any events to that successor before epoch T.”

Advance promises are set via a runtime method (from Node via NodeRuntime, forwarded to the active SimProc).

Rules:

- Once an advance promise is set to epoch `T` for a successor, attempting to send an event to that successor for
  an earlier epoch is a runtime error.
- The SimProc will not promise an epoch earlier than the advance promise for that successor; the successor’s promised
  frontier is advanced accordingly.

Advance promises are useful to explicitly create concurrency windows for successors when the upstream node knows it will
remain idle for a period of simulation time.

### 10.8 Topology constraints and legality of communication

A SimProc may only communicate with declared successors:

- `send_event(target_node, target_simproc, ...)` is legal only if `(target_node, target_simproc)` is an allowed successor
  for this SimProc.
- Promise routing is likewise only along declared successor edges.

Layer ordering constraints (e.g. “cannot send to same or higher layer within the same node”) are enforced by **topology
construction** and by the successor sets delivered to each SimProc. A SimProc itself does not compare layer numbers to
decide legality; it relies on its allowed successors.

Separately, NodeRuntime/Worker may enforce **attention order** across SimProcs as a scheduling policy.

### 10.9 Message ingress: receive_event and receive_promise

A SimProc exposes ingress methods that are invoked by the node/transport controller layer:

- `receive_event(sender_node, sender_simproc, epoch, data, headers)`
- `receive_promise(sender_node, sender_simproc, seqnr, epoch, num_events)`

Ingress rules:

- Events cannot arrive “in the past” relative to the EventQueue’s enabled/processed frontier; past events raise a timing error.
- Promises are applied to the EventQueue; if applying a promise causes the SimProc’s `next_epoch` to become enabled, the
  SimProc may immediately update `next_epoch` and propagate new promises downstream.

### 10.10 Determinism and concurrency properties

A SimProc is designed to support deterministic distributed simulation with concurrency:

- **Deterministic enabling**: Epoch advancement is gated by per-predecessor promises and event counts in the EventQueue.
- **Promises-first transmission**: The outbox ensures promises are routed before events, improving downstream knowledge and
  reducing idle waiting.
- **Out-of-order robustness**: Receiver-side promise rules tolerate transport reordering.
- **Concurrency windows**: Repromises downward, hard wakeups, and advance promises create safe windows where downstream
  nodes can progress independently.

### 10.11 Error conditions and diagnostics

The reference SimProc raises runtime errors for invalid timing and broken progress assumptions, including:

- sending events into the past (`epoch < current epoch`),
- sending events earlier than the last promised epoch to a successor,
- receiving events for an epoch that is already behind the EventQueue frontier,
- reaching a state with no predecessors and no future wakeups (no further handler invocations are possible).

Inspection hooks:

- `waiting_for`: identifies (one of) the predecessors currently blocking progress (useful for debugging deadlock-like stalls).
- string representation includes epoch, next_epoch, queue state, and successor promise state (useful in logs).

---

**Note:** This chapter documents the normative behavior of the reference Python implementation. Performance-oriented
implementations must preserve these semantics, particularly around epoch-0 initialization, promise propagation, and
EventQueue gating rules.


## 11. NodeRuntime and Node

This chapter specifies the **NodeRuntime** and **Node** building blocks and their interaction with **SimProc** and
**EventQueue**. Together, these components define the per-node execution environment used by a Worker to run a
distributed simulation deterministically.

### 11.1 Roles and responsibilities

#### 11.1.1 Node (model logic)

A **Node** is the unit of simulation logic implemented by a **Model**. Disco provides an abstract `Node` base class that
is subclassed by model authors.

A Node:

- receives events (batched per SimProc advancement) through `on_events(simproc_name, events)`,
- updates node state and produces outputs (events, wakeups, advance promises),
- does **not** communicate with transports directly,
- does **not** access SimProcs directly; instead it uses runtime methods provided by NodeRuntime.

Nodes are not part of the Disco engine distribution beyond the abstract base class; concrete node implementations are
provided by the Model package.

#### 11.1.2 NodeRuntime (per-node controller)

A **NodeRuntime** is a per-node controller created by the Worker for each logical node instance.

A NodeRuntime:

- owns the node’s **SimProcs** (exactly one per Graph layer / model simproc),
- owns the node’s **EventQueues** indirectly through SimProcs (exactly one EventQueue per SimProc),
- provides the **Node-facing runtime API** (`send_event`, `wakeup`, `advance_promise`, etc.),
- provides **ingress hooks** for transports (`receive_event`, `receive_promise`),
- maintains the **active SimProc context** while executing the node handler,
- runs a deterministic **attention loop** (`runner(...)`) that advances the node’s SimProcs in the correct order.

### 11.2 Construction and initialization

A NodeRuntime is constructed with the information required to bind together model logic, topology, routing, and
observability:

- `spec: NodeInstanceSpec` identifies the node instance (node name, node type, labels).
- `model: Model` provides the node factory and the ordered list of simproc names.
- `partitioning: Partitioning` provides per-(node, simproc) predecessors and successors.
- `router: Router` routes non-local envelopes.
- `graph: Graph` provides the graph view (including node masks) to model logic via the runtime.
- `dlogger: DataLogger` and `seed` provide instrumentation and stochastic control.

During construction, NodeRuntime:

1. Instantiates the Node implementation via `model.node_factory(node_type, runtime=self)`.
2. Creates one `SimProc` per model simproc name (ordered), binding:
   - the node’s `on_events` handler as the simproc callback,
   - `router.send_event` and `router.send_promise` as routing delegates,
   - predecessor and successor sets for that (node, simproc).
3. Initializes bookkeeping:
   - status (`INITIALIZED`),
   - `_active_simproc = None` until the runner enters ACTIVE state,
   - simproc name → index mapping for ingress dispatch.

#### 11.2.1 Node initialization

Model code may require initialization prior to simulation start. NodeRuntime therefore exposes:

- `initialize(**kwargs)`: forwards to `node.initialize(**kwargs)` and leaves the runtime in `INITIALIZED` state.

The Worker (or higher-level orchestration) is responsible for invoking `initialize` before starting the runner.

### 11.3 NodeRuntimeLike interface and façade semantics

To decouple model code from engine internals, Nodes are constructed with a `NodeRuntimeLike` interface (Protocol).

This interface provides:

- `send_event(target_node, target_simproc, epoch, data, headers=None)`
- `wakeup(epoch, hard=False)`
- `advance_promise(target_node, target_simproc, epoch)`
- introspection properties:
  - `name` (node name),
  - `epoch` (current epoch of the active SimProc, or `None` outside handler execution),
  - `active_simproc_name` and `active_simproc_number`,
  - optionally runtime-provided objects such as `graph`, `dlogger`, and `seed` (implementation-dependent).

#### 11.3.1 Active SimProc context

A NodeRuntime maintains an **active SimProc** while executing model logic.

- When `SimProc.try_next_epoch()` advances and decides to invoke the handler, it calls the bound callback
  `node.on_events(simproc_name, events)`.
- Immediately before that callback, the NodeRuntime must have selected that SimProc as `_active_simproc`.
- During `on_events`, calls made by the Node (`send_event`, `wakeup`, `advance_promise`) are forwarded to the active
  SimProc only.

This provides two essential invariants:

1. Node code never needs a SimProc handle; it always uses runtime methods.
2. All node outputs are unambiguously attributed to the correct *(node, simproc)* stream.

### 11.4 Outgoing communication API

NodeRuntime exposes runtime methods that are safe to call **only** while the node is in `ACTIVE` state.

If called while not active, these methods raise a runtime error.

#### 11.4.1 send_event

`send_event(...)` forwards to `active_simproc.send_event(...)`.

SimProc enforces legality and timing, including:

- events cannot be sent into the past relative to the active SimProc epoch,
- events cannot be sent earlier than the last promised epoch for the successor,
- events can only be sent to declared successor *(node, simproc)* targets.

Events are staged in the SimProc outbox to preserve the **promises-first** discipline: promises are routed before the
corresponding events are sent.

#### 11.4.2 wakeup

`wakeup(epoch, hard=False)` forwards to `active_simproc.wakeup(epoch, hard)`.

Wakeups are SimProc-local. A hard wakeup can postpone earlier epochs and is used to create concurrency windows and
time-based behavior. Every SimProc also has an implicit hard wakeup at epoch 0 (see Chapter 10).

#### 11.4.3 advance_promise

`advance_promise(target_node, target_simproc, epoch)` forwards to `active_simproc.advance_promise(...)`.

Advance promises are per-successor constraints stating that no events will be sent to that successor before the given
epoch. Trying to send an event earlier than an advance promise results in a runtime error.

### 11.5 Ingress API and SimProc dispatch

NodeRuntime provides ingress hooks used by transports or IPC mechanisms:

- `receive_event(EventEnvelope)`
- `receive_promise(PromiseEnvelope)`

#### 11.5.1 Dispatch by target_simproc

Ingress is dispatched strictly by `envelope.target_simproc`:

- the target simproc name is mapped to a local SimProc instance,
- the envelope contents are forwarded to that SimProc via `receive_event` or `receive_promise`.

NodeRuntime relies on transports to deliver envelopes to the correct target node. NodeRuntime itself does not perform
additional target validation in the reference implementation.

#### 11.5.2 Promise sequence numbers

Promise envelopes include `seqnr` to support out-of-order delivery. NodeRuntime forwards the sequence number to the
SimProc / EventQueue promise logic, which applies receiver-side ordering rules.

### 11.6 Runner loop and deterministic attention ordering

NodeRuntime exposes `runner(duration)` as a generator used by the Worker runner. While active, the runner repeatedly
selects one SimProc to receive attention.

#### 11.6.1 Selection policy

The selection policy is deterministic and consists of:

1. **Earliest epoch first**: select the SimProc with the smallest defined `next_epoch`.
2. **Tie-break by simproc order**: if multiple SimProcs have the same `next_epoch`, the SimProc with the highest order
   (priority) is selected.

Priority order is defined by the iteration order of `model.spec.simprocs` and the corresponding `SimProc.number`
assignment. NodeRuntime iterates SimProcs from **higher order to lower order**.

The reference implementation realizes this tie-break without an explicit comparison:

- the active SimProc is updated only when a strictly smaller `next_epoch` is found (`<`),
- therefore, for equal `next_epoch`, the first (highest-order) SimProc encountered remains active.

#### 11.6.2 Waiting for promises and idling

If any SimProc has `next_epoch is None`, the node cannot determine which epoch to process next and must wait for further
promises. In this situation, NodeRuntime:

- sets `_waiting_for` to an inspection message,
- yields control to the Worker runner with a small backoff (`NO_NEWS_SKIP`).

If the active SimProc cannot advance because it is still waiting for events (`try_next_epoch()` returns `False`),
NodeRuntime likewise yields with backoff, updating `_waiting_for` for diagnostics.

#### 11.6.3 Completion

The runner stops when the smallest next epoch across SimProcs is beyond the requested duration:

- if `next_epoch >= duration`, NodeRuntime transitions to `FINISHED` and returns from the generator.

### 11.7 Observability and diagnostics

NodeRuntime provides inspection properties for debugging and observability:

- `epoch`: current epoch of the active SimProc (or `None` outside handler execution),
- `active_simproc_name` / `active_simproc_number`,
- `_waiting_for`: a human-readable string describing what blocks progress (e.g. which predecessor is awaited),
- structured logging at event/promise ingress, event emission, and wakeup/promise operations.

The `DataLogger` and `Graph` objects are exposed to Node implementations via the runtime where applicable, enabling model
instrumentation without direct access to engine internals.

### 11.8 Failure semantics

If Node code raises an exception during `on_events`, the reference implementation may allow the exception to propagate to
the Worker, which can then transition the node to `FAILED` and terminate the simulation (policy-dependent).

Runtime methods (`send_event`, `wakeup`, `advance_promise`) raise a `DiscoRuntimeError` if invoked when the node is not in
`ACTIVE` state, preventing accidental side effects outside handler execution.


## 12. Experiments

This chapter specifies the **Experiment** abstraction used to orchestrate simulation runs consisting of multiple
**replications**, optionally partitioned into parallel **assignments** executed by workers. Experiments are
persisted in the **Metastore** (Chapter 3) and are updated by multiple worker processes concurrently using
**optimistic concurrency** (CAS) and **atomic read–modify–write** operations.

The implementation is intentionally split into:

- **Model (dataclasses)**: `disco.experiments.model`
- **Persistence / orchestration helpers**: `disco.experiments.store`
- **Public package API**: `disco.experiments` (re-exports only the public surface)

### 12.1 Purpose

An Experiment represents a bounded simulation run:

- It has a fixed runtime budget (`duration`) and a `scenario`.
- It is executed multiple times (**replications**) to obtain statistical confidence.
- Each replication may be partitioned into multiple independent execution units (**assignments**) that can be
  scheduled on different workers.
- Workers report progress and failures back to the orchestrator by writing updates to the Metastore.
- The orchestrator and external tooling can monitor experiment progress by reading a single experiment record.

A core requirement is that **experiment updates are atomic** and remain **internally consistent** even when many
processes update the same experiment concurrently.

### 12.2 Data model

#### 12.2.1 ExperimentStatus

Experiment lifecycle is represented by the `ExperimentStatus` enum:

- `CREATED`
- `SCHEDULED`
- `ASSIGNED`
- `LOADED`
- `INITIALIZED`
- `ACTIVE`
- `PAUSED`
- `FINISHED`
- `CANCELED`
- `FAILED`

The status values are ordered but **ordering is not used for comparison**; instead, explicit propagation rules
(Section 12.3) define the aggregate status.

#### 12.2.2 Assignment

An **Assignment** is the smallest unit scheduled to a worker.

Fields:

- `partition: int`  
  The partition index for this assignment. A replication can have one or more partitions.
  If an experiment is **unpartitioned**, partition `0` represents the entire replication.
- `worker: str`  
  Worker address/identifier.
- `status: ExperimentStatus`
- `exc: str | None`  
  Optional exception/traceback message reported by the worker.

Assignments live inside a replication keyed by `partition`.

#### 12.2.3 Replication

A **Replication** represents one run of the scenario.

Fields:

- `repid: str`  
  Unique replication id (UUID string).
- `repno: int`  
  Sequence number within the experiment.
- `seeds: list[int]`  
  List of per-partition seeds. Must contain **at least as many seeds as partitions** used by the replication.
  In unpartitioned mode, at least one seed is required.
- `status: ExperimentStatus`
- `assignments: dict[int, Assignment]`  
  Mapping of partition -> assignment.
- `exc: str | None`  
  Aggregated exception (derived, see Section 12.3).

#### 12.2.4 Experiment

An **Experiment** aggregates all replications.

Fields:

- `expid: str`  
  Unique experiment id (UUID string).
- `duration: float`  
  Runtime budget for the replication runtime (must be > 0).
- `scenario: str`  
  Scenario identifier (string).
- `allowed_partitionings: list[str]`  
  Identifiers for permissible partitioning schemes.  
  **May be empty**, which denotes **unpartitioned mode** (equivalent to a partitioning with one partition).
- `selected_partitioning: str | None`  
  The partitioning chosen by the orchestrator.  
  In unpartitioned mode, this must be `None`.
- `replications: dict[str, Replication]`  
  Mapping of repid -> replication.
- `params: dict[str, Any] | None`  
  Optional scenario parameters.
- `status: ExperimentStatus`
- `exc: str | None`  
  Aggregated exception (derived, see Section 12.3).

### 12.3 Status and exception propagation

Statuses and exceptions propagate upward:

- Assignment → Replication
- Replication → Experiment

The propagation rules are:

1. If **any child is FAILED**, the parent is FAILED.
2. Else if **any child is CANCELED**, the parent is CANCELED.
3. Else if **all children are FINISHED** and there is at least one child, the parent is FINISHED.
4. Else if **any child is ACTIVE**, the parent is ACTIVE.
5. Otherwise, the parent status remains unchanged (the orchestrator may set intermediate states such as
   `SCHEDULED`, `ASSIGNED`, `LOADED`, etc.).

Exception propagation is defined as:

- Prefer the first exception of a **FAILED** child (assignment/replication) if present.
- Otherwise, propagate the first non-empty exception of any child.
- If no exceptions exist, the parent exception is `None`.

#### 12.3.1 Normalization

To avoid inconsistent persisted state, the model provides **normalization** hooks:

- `Replication.normalize()`
- `Experiment.normalize()`

Normalization must be:

- **Idempotent** (multiple calls do not change the result).
- **Deterministic** (depends only on contained data).
- Responsible for recalculating derived fields (aggregate `status` and `exc`) from leaf state.

The persistence layer (Section 12.6) calls `Experiment.normalize()` **before every committed write**.

### 12.4 Partitioning and assignment

Partitioning controls the number of assignments per replication:

- If `allowed_partitionings` is empty:
  - The experiment is **unpartitioned**.
  - A replication has exactly one assignment at `partition=0`.
  - `selected_partitioning` is `None`.
- If `allowed_partitionings` is non-empty:
  - The orchestrator chooses a `selected_partitioning` from the allowed list.
  - The number of partitions is defined by the orchestrator’s partitioning implementation for that identifier.

Assignments are created by the orchestrator using:

- `Experiment.assign_partition(repid, partition, worker, ...)`
- Convenience: `Experiment.assign_replication(repid, worker)` (equivalent to `partition=0`)

Workers report progress by updating assignment status and/or exception:

- `Experiment.set_assignment_status(repid, partition, status)`
- `Experiment.set_assignment_exc(repid, partition, exc)`

### 12.5 Seeds

Seeds are stored per replication to ensure reproducibility and independent randomness per partition.

Rules:

- Each replication’s `seeds` must contain at least as many values as partitions used.
- In unpartitioned mode (one partition), at least **one seed** is required.
- Missing seeds are generated using random **32-bit** unsigned values.

The model provides:

- `Replication.ensure_seeds(n_partitions, seed_factory=...)`

The orchestrator is responsible for determining `n_partitions` for the selected partitioning and ensuring seeds
are present before scheduling execution.

### 12.6 Persistence in the Metastore

#### 12.6.1 Single-key experiment record

Each experiment is stored as a **single value** under a single logical key:

```
/experiments/<expid>
```

Storing the entire experiment as one blob ensures that:

- **Load/store/update happens in one atomic key update**.
- The Metastore can enforce concurrency via CAS (Chapter 3).
- Readers always observe a complete, self-consistent snapshot.

The experiment object is serialized (by default using `pickle`) as a dictionary produced by
`Experiment.to_dict()` and reconstructed by `Experiment.from_dict()`.

#### 12.6.2 Atomic updates under concurrency

Workers and the orchestrator may update the same experiment concurrently from different OS processes.
To prevent lost updates, all modifications must be done using the Metastore’s optimistic concurrency helpers:

- `Metastore.get_key_with_version(path) -> (value, VersionToken | None)`
- `Metastore.compare_and_set_key(path, value, expected=VersionToken) -> bool`
- `Metastore.atomic_update_key(path, updater, ...) -> value`

The experiment layer wraps these primitives in `ExperimentStore`:

- `ExperimentStore.store(experiment)`  
  Writes the full experiment blob.
- `ExperimentStore.load(expid)` / `reload(experiment)`  
  Reads the full experiment blob.
- `ExperimentStore.atomic_update(expid, mutator)`  
  Performs atomic read–modify–write using `Metastore.atomic_update_key`.  
  The store **always calls `Experiment.normalize()` before commit**.

Additionally, convenience methods are provided for the most common updates (e.g., generate replications,
select partitioning, assign partitions, update status/exceptions). These methods are the recommended surface
for worker updates.

### 12.7 Public API

The public API is exposed via `disco.experiments` and intentionally re-exports only the stable surface:

- `ExperimentStatus`
- `Assignment`
- `Replication`
- `Experiment`
- `ExperimentStore`
- `experiment_path`

Internal helpers and private recomputation routines are not part of the public API.

### 12.8 Error handling and invariants

- `duration` must be > 0; invalid experiments are rejected at construction time.
- In unpartitioned mode (`allowed_partitionings == []`), `selected_partitioning` must be `None`.
- `ExperimentStore.load()` raises `KeyError` if an experiment does not exist.
- Atomic updates may fail after repeated contention and raise `MetastoreConflictError` (Chapter 3).

### 12.9 Forward compatibility (etcd)

The concurrency model is designed to migrate from ZooKeeper/Kazoo to etcd with minimal surface change:

- `VersionToken` maps from ZooKeeper `stat.version` to etcd `mod_revision`.
- `compare_and_set_key` maps to etcd transactions (compare mod_revision → put).
- `atomic_update_key` remains a CAS loop with retry and jitter.

The experiment layer depends only on the abstract Metastore API (Chapter 3), not on ZooKeeper specifics.

## 13 Server

This chapter specifies the **Server** component (`src/disco/server.py`): the process supervisor that runs in the
**application main process** and is responsible for spawning and managing one or more **Worker** processes.

The Server is intentionally small and operationally focused. It does **not** implement routing, transports,
node runtimes, or simulation logic. Instead, it wires together:

- **Process supervision**: spawn and reap worker processes (and optionally an orchestrator process).
- **Control plane access**: create a process-local **Cluster** client (Chapter 4) to drive desired-state changes.
- **Local IPC wiring**: create and share the `event_queues` and `promise_queues` maps used by the IPC transport
  (Chapter 6) when multiple workers run inside the same pod or host.

The Server is designed to be **Kubernetes-friendly**:

- It may run multiple workers inside a single pod for efficient IPC.
- When running as PID 1, it handles SIGTERM/SIGINT and cleanly terminates/reaps child processes.

---

### 13.1 Responsibilities and non-goals

**Responsibilities**

1. Determine the worker set (addresses, ports) for this application instance.
2. Create local IPC queues and pass the same queue dictionaries to each worker process.
3. Create a Cluster client in the main process for control-plane operations.
4. Spawn worker processes using `multiprocessing` with a start method that avoids unsafe inheritance of networking
   state (see 13.3 and Chapter 3).
5. Handle termination signals and enforce a bounded, deterministic shutdown sequence.

**Non-goals**

- The Server does not implement the Worker runner loop (Chapter 5).
- The Server does not implement routing or transports (Chapter 6).
- The Server does not change worker state directly; it uses **Cluster desired state** and OS process termination.

---

### 13.2 Public API

The Server is created as:

```python
Server(
    settings: AppSettings,
    *,
    workers: int | None = None,
    ports: list[int] | None = None,
    bind_host: str | None = None,
    group: str | None = None,
    grace_s: int | None = None,
    orchestrator: bool = True,
)
```

and started with:

```python
server.start()  # blocks until all workers have exited
```

Key arguments:

- `settings`: canonical application settings (see `AppSettings` in `config.py`).
- `workers` / `ports`: define how many workers to start and on which ports (Section 13.4).
- `bind_host`: the host part for worker addresses (Section 13.4). Must be routable for multi-pod deployments.
- `group`: ZooKeeper group/namespace passed into `Cluster.make_cluster(...)` (Chapter 3/4).
- `grace_s`: optional override for the SIGTERM grace period; defaults to `AppSettings.grace_s`.
- `orchestrator`: whether to start a placeholder orchestrator process (Section 13.7).

---

### 13.3 Process model and the “one ZooKeeper client per process” invariant

**Invariant (Chapter 3):** there must be **exactly one KazooClient per OS process**, and a KazooClient must never be
shared across processes or inherited via `fork`.

Therefore:

- The **main Server process** creates its own Cluster client via:

  ```python
  with Cluster.make_cluster(zookeeper_settings=settings.zookeeper, group=group) as cluster:
      ...
  ```

- **Each Worker process** creates its own Cluster client via the same context manager inside the worker entrypoint,
  and passes the resulting `cluster` into `Worker(address=..., cluster=cluster, ...)`.

To reduce the risk of unsafe inherited state (sockets, locks, threads), the Server uses a **spawn-based
multiprocessing context** (or an equivalent method that avoids inheriting parent process state). This ensures worker
processes start from a clean interpreter and re-import modules normally.

---

### 13.4 Worker identity, bind host, and port selection

Workers are identified in the Cluster by their **worker address string**:

```
"<bind_host>:<port>"
```

This address is used consistently:

- As the worker identity stored in Cluster metadata (Chapter 4).
- As the gRPC endpoint address for inter-worker communication (Chapter 6).
- As the dictionary key for IPC queues (`event_queues[address]`, `promise_queues[address]`).

#### 13.4.1 Determining worker count

The Server determines the worker count as:

1. If `ports` is provided: **one worker per port**.
2. Else if `workers` is provided: use that value.
3. Else: `max(1, cpu_count() - 1)`.

#### 13.4.2 bind_host requirements

`bind_host` must be routable in the target deployment topology.

- The Server does **not** silently default to `localhost`/`127.0.0.1`, because that produces unusable worker
  addresses in clustered deployments.
- A recommended configuration knob is `GrpcSettings.bind_host`, set via `DISCO_GRPC__BIND_HOST`, typically to the
  **Pod IP** in Kubernetes.
- For local development, passing `bind_host="127.0.0.1"` explicitly is allowed.

If no `bind_host` can be determined safely, the Server raises `ConfigError` with guidance to set
`DISCO_GRPC__BIND_HOST`.

#### 13.4.3 Port selection

- If `ports` is provided, the Server uses them directly.
- Otherwise, it selects ports by asking the OS for free ports (`bind((bind_host, 0))`) and then assigning those
  ports to workers.

Note: free-port discovery has a small race window. In practice, within a single pod starting workers immediately,
this is acceptable. For more controlled deployments, explicit `ports` is preferred.

---

### 13.5 IPC queue topology for multi-worker pods

When multiple workers run in the same pod/host, Disco can route envelopes via **IPCTransport** (Chapter 6).
The Server sets this up by creating local queues in the main process:

- `event_queues: dict[str, multiprocessing.Queue]`
- `promise_queues: dict[str, multiprocessing.Queue]`

Both dictionaries are keyed by worker address and contain one queue per worker:

```python
for address in worker_addresses:
    event_queues[address] = ctx.Queue()
    promise_queues[address] = ctx.Queue()
```

The Server then passes the **same dictionary objects** into each worker process:

- Workers can put envelopes into the destination worker’s queue based on the address book.
- Each worker consumes from its own queues.

This design minimizes overhead in multi-worker pods and keeps the routing logic inside the Worker/Transport layer.

---

### 13.6 Singleton guard

Within a single application process, only one Server instance may be running at a time.

The Server enforces a **process-global singleton guard** to prevent accidental double-supervision (for example, when
an application entrypoint is invoked twice or imported incorrectly). On violation, `Server.start()` raises a
`RuntimeError`.

---

### 13.7 Optional orchestrator process (placeholder)

The Server may spawn an **orchestrator process** when `orchestrator=True`. Today, this is a placeholder that:

- Starts as a separate OS process.
- Receives a `multiprocessing.Event` (`stop_event`).
- Waits until the stop event is set, then exits.

In a future implementation, the orchestrator will likely participate in control-plane coordination
(experiments/replications/partition assignment). When implemented, it must follow the same Chapter 3 invariant and
create its **own** Cluster client via `Cluster.make_cluster(...)`.

---

### 13.8 Signal handling and shutdown semantics

The Server must behave correctly as PID 1 in a container.

#### 13.8.1 Signal handlers

The Server installs handlers for:

- `SIGTERM` (Kubernetes termination)
- `SIGINT` (local development)

The handlers set an internal shutdown flag. The main loop performs the actual shutdown logic.

#### 13.8.2 Cooperative shutdown via desired state

On SIGTERM, the Server first attempts a cooperative shutdown:

1. For each worker address, call:

   ```python
   cluster.set_desired_state(worker_address=address, state=WorkerState.TERMINATED)
   ```

2. Wait up to `grace_s` for workers to exit.

Rationale:

- Workers are expected to observe desired state changes and stop in their runner thread (Chapter 5).
- Desired state is the primary control-plane contract (Chapter 4).

#### 13.8.3 Escalation (process-level enforcement)

Desired-state may not be processed if a worker is hung (deadlock, stuck syscall, bug). Therefore, the Server must
**not** rely solely on desired state to enforce termination.

After the grace period expires, the Server escalates:

1. Call `Process.terminate()` on remaining alive worker processes.
2. Wait a short additional window.
3. Force kill remaining processes (`Process.kill()` if available; otherwise platform `SIGKILL`).
4. Join/reap children.

This escalation sequence is required to guarantee bounded termination in Kubernetes.

---

### 13.9 Kubernetes operational guidance

Recommended practices:

- **Bind host**: set `DISCO_GRPC__BIND_HOST` to a routable address (typically the Pod IP via the Downward API).
- **Grace periods**:
  - Configure `terminationGracePeriodSeconds` in the Pod spec.
  - Set `AppSettings.grace_s` (and optionally the Server `grace_s` override) to fit within that window.
- **Process model**:
  - Run the Server as the container entrypoint.
  - Do not use daemon child processes; the Server must own and reap worker PIDs.
- **Multi-worker pods**:
  - Use multiple workers per pod when local IPC is beneficial.
  - Ensure CPU and memory limits match the configured worker count.

---

### 13.10 Testing strategy

Server behavior is tested primarily with **unit tests** that patch:

- `Cluster.make_cluster(...)` to avoid ZooKeeper.
- Worker construction/`run_forever()` to avoid running a real worker loop.
- The multiprocessing context to avoid spawning real OS processes.

Integration tests (optional) may spawn real processes, but should still mock ZooKeeper and should be marked as slow
or platform-dependent.

## 14. Orchestrator

### 14.1 Purpose and Responsibilities

The **Orchestrator** is the scheduling component that turns *submitted* replications into *assigned* replications and then triggers workers to start them.

It is deliberately **scheduling-only**:
- It **does**:
  - dequeue submissions in FIFO order
  - select a partitioning scheme (once per experiment)
  - wait for sufficient worker availability for an *all-at-once* assignment
  - persist assignments (which sets replication status to **ASSIGNED**)
  - request worker desired states (**READY → ACTIVE**) to start execution
- It **does not**:
  - manage execution progress
  - update statuses beyond **ASSIGNED** (handled by Workers)
  - do locality-aware placement (future work)

Workers are responsible for advancing statuses after assignment:
- **LOADED**, **INITIALIZED**, **ACTIVE**, **PAUSED**, **FINISHED**, **CANCELED**.

### 14.2 Leader Election and Single Active Scheduler

Multiple orchestrator instances may run, but only the elected leader performs scheduling.

- The orchestrator uses `Cluster.make_orchestrator_election(address=...)`.
- `run_forever()` enters leader election; the leader callback is `_on_lead()`.
- On shutdown, `request_stop()` cancels the election and joins any active launch threads.

### 14.3 Submissions Queue and Locking Semantics

Replications are submitted via a metastore-backed queue (ZooKeeper LockingQueue semantics), consumed via:

- `ExperimentStore.dequeue(...) → Submission | None`
- `Submission` is a `QueueEntity` with:
  ```python
  @dataclass(frozen=True, slots=True)
  class Submission(QueueEntity):
      value: tuple[str, str]  # expid, repid
  ```

The queue provides **lock-and-return** behavior:
- `dequeue()` locks the head element and returns it as a `QueueEntity`.
- The item is not removed until `consume()` is called.
- If the orchestrator stops *before* committing to execution, it can call `release()` to make the head available again.

The orchestrator uses this to support clean shutdown without losing submissions:
- **Before handover to launch**: keep locked; on stop, `release()` so it remains available at the head.
- **After assignments are persisted**: call `consume()` **before** spawning the launch thread to prevent retries of corrupt/partially-started replications.

**Important policy:** once assignments have been persisted (status becomes ASSIGNED), the submission is consumed unconditionally. If anything later goes wrong while requesting worker start, the replication is marked failed rather than re-queued.

### 14.4 High-Level Control Flow

Leader loop (`_on_lead()`):

1. Dequeue a submission with a short timeout (polling style):
   - `entity = store.dequeue(timeout=DEQUEUE_TIMEOUT_S, force_mode="raise")`
2. If no item, continue.
3. Process submission:
   - `_handle_submission(entity)`
4. Stop handling:
   - If `_StopRequested`: `release()` and exit leadership loop.
5. Unexpected error:
   - mark replication failed
   - `consume()` to prevent infinite head-of-queue retries

This ensures the queue remains FIFO and the orchestrator never skips ahead.

### 14.5 Submission Handling and Assignment Commit Point

`_handle_submission(entity)` performs:

1. Load experiment metadata: `exp = store.load(expid)`
2. Select partitioning if not yet selected:
   - `partitioning_id = _select_partitioning_for_experiment(exp)`
   - persisted via `store.select_partitioning(expid, partitioning_id)`
3. Determine `num_partitions` from `Partitioning.load_metadata(meta, partitioning_id)`
4. Ensure seeds exist for the replication:
   - `store.ensure_replication_seeds(expid, repid, num_partitions)`
5. Wait for a full assignment plan (all partitions at once):
   - `assignments = _await_full_assignment_plan(expid, num_partitions)`
6. Persist the assignment mapping:
   - `exp = store.assign_partitions(expid, repid, assignments)`
   - **This call also transitions replication status to `ASSIGNED`.**
7. **Commit point:** consume the queue entity:
   - `entity.consume()`
8. Start launch in a daemon thread:
   - `Thread(target=_launch_replication, args=(exp, repid), daemon=True).start()`

**Rationale for consuming before launch:** launch runs in a different thread and the leader cannot reliably determine whether the start sequence completed correctly; re-queuing could trigger duplicate or inconsistent execution. The conservative policy is “no retry; mark failed if start cannot be confirmed”.

### 14.6 Partitioning Selection Policy

Partitioning is selected **once per experiment** based on:
- `Experiment.allowed_partitionings` (ordered list)
- cluster size and availability

Policy in `_select_partitioning_for_experiment(exp)`:
- Iterate allowed schemes in order.
- Skip schemes where `total_workers < num_partitions` (can never run).
- Prefer the first scheme that is runnable immediately with current AVAILABLE workers.
- Otherwise, if total cluster size is sufficient, wait briefly (`PARTITIONING_FALLBACK_S`) for availability.
- If still not runnable, consider the next scheme (typically fewer partitions).

This keeps the implementation simple and predictable; more advanced heuristics can be added later.

### 14.7 All-at-Once, Reuse-Aware Placement

Assignments are computed in `_await_full_assignment_plan(expid, num_partitions)`:

- It **never** returns a partial plan.
- It waits until there are at least `num_partitions` AVAILABLE workers.

Reuse awareness:
- `Cluster.get_available(expid)` returns:
  - `addresses`: AVAILABLE workers ordered by preference
  - `preferred_partitions`: partition indices already set up on preferred workers
- The planner:
  - preserves compatible preferred bindings first (distinct partitions in range)
  - fills remaining partitions with remaining workers in FIFO order

Output is an ordered list of workers by partition index:
- `ordered_workers[p] = worker_for_partition_p`

### 14.8 Launch Thread and Worker Start Protocol

`_launch_replication(exp, repid)` requests worker desired states and waits for confirmation via observed worker states:

1. Request READY for all assigned workers:
   - `Cluster.set_desired_state(..., state=READY, expid, repid, partition)`
2. Barrier wait until all assigned workers report `WorkerState.READY`
   - bounded by `settings.launch_timeout_s`
   - polled using `Cluster.await_available(timeout=READY_POLL_S)`
3. Request ACTIVE for all partitions:
   - `Cluster.set_desired_state(..., state=ACTIVE, ...)`
4. Barrier wait until all assigned workers report `WorkerState.ACTIVE`
   - also bounded by `settings.launch_timeout_s`

Failure policy:
- If orchestrator stop is requested during launch, mark replication failed.
- If a timeout occurs, mark replication failed.

**Note:** the orchestrator does not set statuses beyond `ASSIGNED`. Workers update statuses when they complete loading/initializing/starting.

### 14.9 Shutdown Semantics

`request_stop(timeout_s: Optional[float])`:
- sets `_stop`
- cancels leader election
- joins launch threads:
  - bounded if `timeout_s` is provided and > 0
  - otherwise waits until all launch threads complete

Leader loop shutdown behavior:
- If stop happens while a queue item is locked but not yet committed:
  - `_StopRequested` is raised
  - `entity.release()` is called
  - leader loop exits cleanly
- If stop happens after commit (entity consumed):
  - the launch thread is responsible for marking failed if it cannot complete start sequence

### 14.10 Error Handling and “Fail Closed” Policy

The orchestrator treats unexpected scheduling errors as potentially non-retriable:
- it marks replication failed (with an exception payload for diagnostics)
- it consumes the submission to prevent repeated head-of-queue failures

This avoids queue deadlocks and prevents “corrupt” replications from being retried indefinitely.

### 14.11 Testing Strategy

Orchestrator tests should avoid real ZooKeeper:
- queue behavior is tested independently in metastore tests using a queue factory / fake locking queue
- orchestrator tests use fakes/mocks for:
  - `ExperimentStore` (dequeue/load/assign/mark-failed)
  - `Cluster` (availability, desired state, worker state transitions)
  - leader election (invoke `_on_lead()` synchronously)

Key scenarios:
- FIFO dequeue and commit: consume before launching
- stop while waiting for availability: release and exit
- unexpected scheduling exception: mark failed and consume
- launch timeouts: mark failed (without re-queue)

## 15. Client

### 15.1 Purpose and Responsibilities

The **Client** is the user-facing entry point for interacting with a Disco cluster.

It is responsible for:

- **Submitting work** to the cluster by enqueueing replications via `ExperimentStore.submit(...)`.
- **Monitoring** cluster and experiment state using read-only operations on `Cluster` and `ExperimentStore`.
- **Issuing control requests** such as **pause** and **cancel**, which are expressed as *desired worker state* updates.
- **Optionally** performing direct worker inspection over **gRPC** (when enabled) for diagnostics.

The Client is **not** responsible for scheduling decisions, partition selection, or assignment planning. Those are handled by the **Orchestrator**.


### 15.2 Dependencies and Design Constraints

This chapter follows the architecture of:

- **Cluster** (Chapter 4): in-memory view of the worker fleet backed by ZooKeeper watches via `Metastore`.
- **Experiments** (Chapter 12): experiments are stored as a single znode blob and updated atomically.

Key constraints:

- Experiment submissions are done via `ExperimentStore.submit(expid, repid)`.
- The Client does **not** use a “Cluster gRPC client”; gRPC (if used) is **directly** from Client → Worker.
- Worker-driven lifecycle: Workers transition replication/partition status beyond **ASSIGNED** (e.g., `LOADED`, `INITIALIZED`, `ACTIVE`, `PAUSED`, `FINISHED`, `CANCELED`).


### 15.3 Construction and Lifecycle

The Client owns the ZooKeeper connection lifecycle when constructed via `Client.make(...)`.

#### 15.3.1 `Client.make(...)`

**Signature:**

- `Client.make(settings: AppSettings | None = None, group: str | None = None) -> Client`

**Behavior:**

- If `settings` is not provided, it is obtained from `disco.config.get_settings()`.
- A `ZkConnectionManager` is created and started.
- A `Metastore` is created using the chosen `group` (defaulting to `settings.zookeeper.default_group`).
- A `Cluster` and `ExperimentStore` are created from the shared `Metastore`.
- The Client returns an instance that is ready for use.

The Client should implement `close()` and be usable as a context manager:

- `Client.__enter__` returns `self`.
- `Client.__exit__` calls `close()`.

`close()` stops the underlying `ZkConnectionManager`. After closure, operations should raise a runtime error.


### 15.4 Core API

#### 15.4.1 Submitting Experiments

**Method:** `submit(exp: str | Experiment, repid: str | None = None) -> None`

Semantics:

- If `exp` is an `Experiment`, the Client first stores it via `ExperimentStore.store(exp)`.
- If `exp` is a string, it is interpreted as `expid`.
- If `repid` is provided, submit exactly that replication using `ExperimentStore.submit(expid, repid)`.
- If `repid` is omitted (`None`), submit **all replications** of the experiment:
  - Load the experiment (`ExperimentStore.load(expid)`), iterate over `exp.replications.keys()`, and submit each.

Notes:

- Submission is an enqueue-only operation; there is no guarantee that the Orchestrator is currently leading.
- Submitting the same `(expid, repid)` multiple times is allowed by the queue; client code should avoid duplicates unless explicitly desired.


#### 15.4.2 Cluster Inspection

Convenience inspection methods expose the **current in-memory view** from `Cluster`.

- `list_workers() -> Mapping[str, WorkerState]`
  - Returns worker address → current `WorkerState`.

- `get_worker_info(worker: str) -> WorkerInfo`
  - Returns the worker’s `WorkerInfo` as stored under `/simulation/workers/<worker>`.

- `get_active_orchestrator() -> LeaderRecord | None`
  - Returns the currently elected orchestrator leader record (or `None`).


#### 15.4.3 Experiment Inspection

The Client offers read-only status summaries suitable for CLI/UX.

- `get_experiment(expid: str) -> Experiment`
  - Loads and returns the experiment blob.

- `list_experiments() -> Mapping[str, ExperimentStatus]`
  - Enumerates known experiments and returns `expid -> status`.
  - Enumeration is performed by listing children under `EXPERIMENTS_ROOT` and loading each experiment.

- `list_replications(expid: str) -> Mapping[str, ExperimentStatus]`
  - Returns `repid -> status` for all replications in the experiment.

These are **snapshots** of ZooKeeper state at read time.


### 15.5 Control Actions: Pause and Cancel

The Client can request a state transition by writing **desired worker state** for all workers assigned to an experiment/replication.

The general flow is:

1. Load the experiment.
2. Determine the target replication(s).
3. Extract assigned workers from the replication’s assignment map.
4. For each worker, call `Cluster.set_desired_state(...)` with the requested state.

#### 15.5.1 `pause(...)`

- `pause(expid: str, repid: str | None = None) -> None`

Requests `WorkerState.PAUSED` for all workers participating in the selected replication(s).

#### 15.5.2 `cancel(...)`

- `cancel(expid: str, repid: str | None = None) -> None`

Requests a termination state for workers running the selected replication(s).

Policy note:

- The exact desired state used for “cancel” depends on the worker contract. Typical choices are `TERMINATED` or a dedicated cancel state (if introduced later).
- The Worker is responsible for reflecting cancellation in experiment status (`CANCELED`) once it has acted on the desired state.

Edge cases:

- If a replication is **not yet ASSIGNED**, there may be no workers to signal. In that case the Client may:
  - (current minimal behavior) do nothing and rely on future Orchestrator logic to skip canceled replications, or
  - (future improvement) atomically mark the replication `CANCELED` in the experiment blob.


### 15.6 Optional Worker Diagnostics via gRPC

The Client may optionally connect directly to workers via gRPC for diagnostics.

Design goals:

- gRPC is **optional** (feature is enabled only when configured and dependencies are present).
- There is no gRPC coupling in `Cluster`.

Examples of future diagnostics:

- Query current simulation epoch.
- Query current “waiting_for” or barrier/debug state.
- Fetch worker-local logs or recent exceptions.

Implementation guidance:

- Client maintains a per-worker channel cache keyed by worker address.
- Failure to connect must not break core Client functionality; gRPC diagnostics should degrade gracefully.


### 15.7 Error Handling and Consistency

- Read operations are best-effort snapshots from ZooKeeper.
- Submissions and desired-state writes are expected to be reliable and cheap; failures should surface as exceptions.
- Client methods should prefer raising clear exceptions (`KeyError` for missing experiment/replication, `RuntimeError` for closed client, etc.).

The Client intentionally avoids complex retry semantics. If an operation fails due to transient ZooKeeper issues, the caller can retry.


### 15.8 Testing Strategy

Client unit tests should focus on:

- Correct usage of `ExperimentStore.submit(...)` for single replication and “submit all”.
- Correct mapping extraction for `list_workers`, `list_experiments`, `list_replications`.
- Correct desired-state writes for `pause` and `cancel` when assignments exist.

Tests should use fakes for `Cluster` and `ExperimentStore` and avoid requiring a live ZooKeeper instance.

## 16. Test Run

### 16.1 Purpose

`disco.testrun.TestRun` is an “elementary Worker” used for debugging, unit tests, and small single-process experiments.

It deliberately removes the distributed control plane:

- No Metastore and no Cluster registration
- No worker addressing, no IPC, no gRPC
- No threading and no runner-control state machine
- No ingress queues (events/promises are delivered in-process)

The intent is to make it as easy as possible to reproduce and debug model behavior in a deterministic, single-process environment while still using the real `NodeRuntime` implementation.

### 16.2 Inputs and Validation

`TestRun` is constructed from:

- `Experiment` (in-memory object; status is updated directly on this object)
- `Graph` (already loaded for the target scenario)
- `AppSettings` (needed for database access and optional model loading)
- Optional: `Model` (if already constructed elsewhere)
- Optional: `DataLogger` (if the caller wants to provide a specific logger instance)
- Optional: `Partitioning` (must describe exactly one partition)

Validation happens in the constructor so failures are immediate and explicit:

- `Partitioning.num_partitions == 1`
- `Partitioning.node_specs` is non-empty
- The `Partitioning.node_specs` order is treated as authoritative for determinism (see below)

If no Partitioning is provided, `TestRun` creates a trivial “single partition” partitioning (via `SimplePartitioner`).

### 16.3 Determinism and Seeding

Determinism goals for `TestRun`:

Given the same:

- `Graph` (scenario data)
- `Experiment.params`
- `Partitioning` (including the order of `node_specs`)
- `master_seed`

…the run should be deterministic within a given environment.

Seed policy:

- The caller provides `master_seed` as either an `int` or a NumPy `SeedSequence`.
- If an `int` is provided, `TestRun` constructs `SeedSequence(master_seed)`.
- `TestRun` spawns one child `SeedSequence` per node: `ss.spawn(n_nodes)`.
- Child SeedSequences are assigned to nodes in the exact order of `partitioning.node_specs`.

Each `NodeRuntime` receives its own `SeedSequence`. The master seed is never passed directly to nodes.

### 16.4 Lifecycle and Status Updates

`TestRun` has a small lifecycle that mirrors the Worker’s “run setup” and “active stepping” but without the control plane.

- Constructor:
  - builds `Router + InProcessTransport`
  - constructs `NodeRuntime` objects for all `node_specs`
  - sets `experiment.status = LOADED`

- `initialize()`:
  - calls `NodeRuntime.initialize(**experiment.params)` for each node in `partitioning.node_specs` order
  - sets `experiment.status = INITIALIZED`

- `run(duration)`:
  - ensures initialization (calls `initialize()` if needed)
  - sets `experiment.status = ACTIVE`
  - creates one runner generator per node for the requested duration
  - steps the runners round-robin until they all raise `StopIteration` for that duration

Important: `TestRun` does not tear down. NodeRuntimes keep their internal state and can be stepped again by calling `run()` again (typically with the same or a later duration).

### 16.5 Error Handling

Unlike the distributed Worker, `TestRun` does not attempt to recover:

- Exceptions are not caught.
- There is no FAILED or BROKEN state for `TestRun`.
- Any error is raised directly to the caller to support debugging.

### 16.6 Relationship to Worker

`TestRun` exists to share the same core execution semantics as a Worker, but without:

- desired-state handling
- Cluster state publishing
- metastore-backed status persistence (ExperimentStore)
- remote transports and ingress queues
- threading/locking concerns

Over time, the Worker should follow the same deterministic “node order” and per-node seed policy as `TestRun` so that a failing distributed run can be reproduced locally with minimal differences.

## 17. Command line interface

### 17.1 Goals

The Disco CLI provides a stable and extensible command-line interface for operating Disco services and running control-plane
actions (e.g. starting a server, queueing experiments, querying cluster state). The design goals are:

- **Pydantic v2 compatible**: all CLI inputs are validated with `BaseModel.model_validate(...)`.
- **Low dependency risk**: avoid third-party argparse/Pydantic glue that may lag behind Pydantic releases.
- **Single source of truth**: CLI flags and help text are derived from the Pydantic command models.
- **Extensible**: adding new subcommands (e.g. client actions) should be a small, local change.

The CLI intentionally does **not** expose the full `AppSettings` surface as flags. Instead, runtime configuration comes from
`get_settings(...)` (env, dotenv, secrets, config file, defaults), while the CLI focuses on top-level operational actions.

---

### 17.2 Entry points

Disco exposes two primary CLI entry points:

1. **Module entry point**: `python -m disco ...`
   - Implemented in `src/disco/__main__.py`.
   - Provides subcommands (e.g. `disco server ...`).

2. **Console script**: `disco-server ...`
   - Implemented in `src/disco/cli/server.py` (or a thin `server_entry.py` wrapper).
   - Starts a Server directly with `ServerCommand`.

The console script is registered via `pyproject.toml`:

```
[project.scripts]
disco-server = "disco.cli.server:main"
```

---

### 17.3 Command models

Each subcommand is described by a Pydantic model. The model defines:

- Field names → CLI flag names (snake_case → `--kebab-case`)
- Types → parsing behavior and validation
- Defaults/Optionals → required/optional flags
- `Field(description=...)` → help text

Example (simplified):

```
class ServerCommand(BaseModel):
    group: str = Field(description="The group for the Server to run in.")
    workers: Optional[int] = Field(None, description="Number of workers to start.")
    ports: Optional[list[int]] = Field(None, description="Ports to run servers on.")
    bind_host: Optional[IPvAnyAddress] = Field(None, description="Bind host.")
    grace_s: Optional[int] = Field(None, description="Grace duration for shutdown.")
    orchestrator: bool = Field(True, description="Disable orchestrator if false.")
    config_file: Optional[str] = Field(None, description="Optional Disco config file (toml/yaml).")
    loglevel: Optional[Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG",...]] = Field(
        None, description="Logging level override."
    )
```

---

### 17.4 Argparse generation from Pydantic models

Disco uses the standard library `argparse` for argument parsing, but builds parsers **generically** from Pydantic model
definitions using `disco.cli.argparse_model.add_model_to_parser(...)`.

This function:

- Iterates over `model.model_fields`.
- Creates argparse arguments from annotations and defaults.
- Uses `Field(description=...)` for `--help` text.
- Parses values conservatively (strings for complex Pydantic types) and relies on `model_validate(...)` for final validation.

Supported patterns:

- `Optional[T]`: optional argument
- `bool`: `--flag` / `--no-flag` via `BooleanOptionalAction`
- `list[T]`: space-separated values via `nargs="*"`
- `Literal[...]`: `choices=...` in argparse (plus Pydantic validation)
- Pydantic types (e.g. `IPvAnyAddress`): parsed as `str`, validated by Pydantic

Parsing flow:

1. `argparse` parses CLI → produces a namespace.
2. `vars(namespace)` is validated by Pydantic: `CommandModel.model_validate(vars(ns))`.
3. The validated model is passed to the command handler.

This approach keeps CLI behavior aligned with command model changes, while maintaining compatibility with Pydantic v2.

---

### 17.5 Server command

The `server` command wires CLI inputs into the `Server` supervisor:

- Settings are loaded through `get_settings(config_file=..., **overrides)`.
- CLI log level (if provided) is applied as a settings override, preserving the settings precedence rules.
- The Server is created and started:

```
server = Server(
    settings=settings,
    workers=command.workers,
    ports=command.ports,
    bind_host=str(command.bind_host) if command.bind_host is not None else None,
    group=command.group,
    grace_s=command.grace_s,
    orchestrator=command.orchestrator,
)
server.start()
```

The Server itself is responsible for Kubernetes-friendly process supervision and for configuring multi-process logging in the
parent process (see Chapter 13).

---

### 17.6 Extending the CLI

To add a new CLI action (e.g. client operations such as queueing experiments):

1. Create a new Pydantic command model, e.g. `QueueCommand`.
2. Implement a handler function `handle_queue(command: QueueCommand)`.
3. Register a new subparser in `disco.__main__.py` (or a dedicated `cli/main.py` if introduced later) using
   `add_model_to_parser(sub.add_parser("queue"), QueueCommand)`.
4. In `main()`, dispatch to `handle_queue(...)`.

Because command models are responsible for help text and validation, extending the CLI typically requires only local,
mechanical changes.

---

### 17.7 Testing

CLI behavior is tested under `tests/cli/`:

- `tests/cli/test_argparse_model.py` verifies parser generation from models:
  required flags, list parsing, boolean toggles, Literal choices.
- `tests/cli/test_cli_entrypoints.py` verifies entrypoints call the correct handlers for:
  - `disco server ...`
  - `disco-server ...`

The CLI tests mock command handlers to avoid starting real worker processes or connecting to ZooKeeper.


---
# Appendices

## A. Naming conventions

This appendix defines the naming conventions for indices and keys across the simulation model and graph schema.

### A.1 Indices

We distinguish between **owning tables** (where the row *is* the entity) and **referencing tables** (where the row refers to that entity).

- **Owning tables (row *is* a vertex / entity)**  
  - The integer index column is named:  
    - `index`  

- **Referencing tables (row refers to a vertex / entity)**  
  - The foreign-key column to the vertex index is named:  
    - `vertex_index`  

Examples:

- `graph.vertices`
  - `scenario_id`
  - `index` – primary integer index for the vertex
- `graph.vertex_labels`
  - `scenario_id`
  - `vertex_index` – FK → `graph.vertices.index`
- `graph.edges`
  - `scenario_id`
  - `source_index`, `target_index` – FKs → `graph.vertices.index`

The same pattern applies to other entity types: an owning table uses a generic `index` column, and referencing tables use `<entity>_index` (for vertices: `vertex_index`).

### A.2 Keys

For business-level (string) identifiers:

- **Owning tables (row *is* a vertex / entity)**  
  - The business key column is named:  
    - `key`

- **Referencing tables (row refers to a vertex / entity)**  
  - The foreign-key column to the business key is named:  
    - `<entity>_key` (for vertices: `vertex_key`)

Examples:

- Node-type data table (owning vertex data for a node type):
  - `scenario_id`
  - `key` – vertex key
- Edge data table (referencing vertices by key):
  - `scenario_id`
  - `source_key`, `target_key` – vertex keys for edge endpoints
