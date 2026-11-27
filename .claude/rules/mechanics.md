---
paths:
  - "src/disco/simproc.py"
  - "src/disco/event_queue/**"
  - "src/disco/runtime.py"
  - "src/disco/node.py"
  - "src/disco/envelopes.py"
  - "src/disco/router.py"
  - "src/disco/partitioning.py"
  - "src/disco/model/spec.py"
---

# The event and promise mechanics are the owner's (decision 0003)

These files hold disco's core: how events and promises flow, how a SimProc
advances its epoch, and the topology that says who may send to whom. Their
behaviour has been misread before, and changes made on that misreading broke
it.

- Change nothing in how events, promises, repromises, wakeups, advance
  promises, epoch advancement, the runner's attention order or the
  predecessor/successor topology behave without the owner's explicit consent,
  recorded in your brief. A refactor that "keeps the behaviour" counts too:
  ask first.
- Before you touch these files, read `ENGINEERING.md` Ch 10 (SimProc) and
  Ch 11 (NodeRuntime) in full, and the docstring of `simproc.py`. The
  reference implementation is normative.
- Key facts that are easy to get wrong:
  - promises are made automatically by the SimProc from the events it sends
    and its `next_epoch`; `advance_promise` is fine control for the Model
    Developer, rarely needed;
  - every (node instance, simproc) pair has an implicit hard wakeup at epoch
    0;
  - within a layer, the node instances form a DAG; across simprocs a node
    sends only from a higher to a lower simproc (self-relations), and
    there are no cross-node, cross-layer edges. The dependency graph over
    (node instance, simproc) pairs is therefore acyclic, which is what makes
    conservative synchronisation work with zero lookahead.
- If a task seems to need a change here, stop and escalate (CLAUDE.md).
