# Routing and Transport layer (Chapter 6)

## Architecture overview

The Router owns an ordered list of Transports and dispatches each outgoing
envelope to the first transport whose `handles_node(repid, node)` returns True.
Priority order: InProcessTransport → IPCTransport → GrpcTransport.

Transports are long-lived: constructed once per Worker, reused across runs.
Do NOT add per-message connection/channel setup — channels are cached in
`_get_or_create_endpoint(addr)`.

## Transport interface (transports.base)
```python
class Transport(Protocol):
    def handles_node(self, repid: str, node: str) -> bool: ...
    def send_event(self, envelope: EventEnvelope) -> None: ...
    def send_promise(self, envelope: PromiseEnvelope) -> None: ...
```

- `handles_node` must be pure and fast (no I/O).
- `send_event` / `send_promise` may raise; the caller (NodeRuntime) decides
  whether to retry or treat failure as fatal.
- Payloads are serialized exactly once in NodeRuntime before reaching the router.
  Transports must not double-serialize.

## Envelope types (Chapter 2)

- EventEnvelope: repid, target_node, target_simproc, epoch, data (bytes), headers
- PromiseEnvelope: repid, target_node, target_simproc, seqnr, epoch, num_events

## Proto service — transport.proto
```proto
service DiscoTransport {
  rpc SendEvents(stream EventEnvelopeMsg) returns (TransportAck);
  rpc SendPromise(PromiseEnvelopeMsg) returns (TransportAck);   // currently unary
}
```

Both EventEnvelopeMsg and PromiseEnvelopeMsg carry `repid` explicitly so the
ingress can reconstruct full envelopes without external context.

## GrpcTransport (egress)

- `_get_or_create_endpoint(addr)` returns a cached `_RemoteEndpoint` (channel + stub).
- `send_event`: opens a short-lived client-streaming call carrying a single message
  (known limitation — no persistent stream yet).
- `send_promise`: unary call with a configurable retry policy:
    - `promise_retry_delays_s: list[float]` — backoff sequence
    - `promise_retry_max_window_s: float` — total retry window
  Retries on RESOURCE_EXHAUSTED or UNAVAILABLE only. Non-retryable errors surface
  immediately.

## GrpcSettings (config.py)

Relevant fields for transport:
- `bind_host`, `bind_port`, `timeout_s`, `max_workers`, `grace_s`
- `compression`
- `promise_retry_delays_s`, `promise_retry_max_window_s`

When adding new transport parameters, add them here as Pydantic fields.

## gRPC Ingress — DiscoTransportServicer

- Receives envelopes from remote workers.
- Does NOT talk to NodeRuntime directly.
- Converts protobuf messages → IPCEventMsg / IPCPromiseMsg and puts them on
  local queues (event_queue, promise_queue).
- Ingress state-gating is NOT done here — it is handled by the Worker runner loop
  when draining its ingress queues.

## Ingress path (end to end)

Remote GrpcTransport
  → DiscoTransportServicer (gRPC thread)
    → event_queue / promise_queue (multiprocessing.Queue)
      → Worker runner loop (drains on each ACTIVE hot-path cycle)
        → NodeRuntime.receive_event / receive_promise

## WorkerState ingress rules (Chapter 2.6)

Ingress is accepted only when Worker is in: READY, ACTIVE, PAUSED.
All other states (CREATED, AVAILABLE, RESERVED, INITIALIZING, TERMINATED,
EXITED, BROKEN) reject or ignore ingress. The gRPC servicer itself does not
enforce this — enforcement happens in the runner loop.