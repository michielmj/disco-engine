Read Chapter 6 of ENGINEERING_SPEC.md in full before starting.

## Task: convert promise transport from unary to client-streaming

Currently `SendPromise` in `DiscoTransport` is a unary RPC. Convert it to a
client-streaming RPC mirroring the `SendEvents` pattern, while preserving the
existing retry semantics.

Scope of changes:

1. **transport.proto**: change `SendPromise` to
   `rpc SendPromise(stream PromiseEnvelopeMsg) returns (TransportAck)`.
   Regenerate stubs after editing the proto.

2. **GrpcTransport.send_promise** (egress): open a short-lived stream
   (single-message, same pattern as send_event) using the cached endpoint.
   Preserve the retry loop from GrpcSettings — wrap the stream call in the
   same retry policy.

3. **DiscoTransportServicer.SendPromise** (ingress): change signature from
   unary to `(self, request_iterator, context)` and iterate over the stream,
   putting each PromiseEnvelopeMsg onto promise_queue as an IPCPromiseMsg.

4. **GrpcSettings**: check whether any config field names need updating.

5. **ENGINEERING_SPEC.md Ch 6**: update the proto snippet and the send_promise
   description to reflect the new streaming signature.

6. **Tests**: update all tests for GrpcTransport.send_promise and
   DiscoTransportServicer.SendPromise to use streaming.

Do not change the retry logic semantics — only the wire transport changes.
Do not touch send_event or SendEvents (already streaming).