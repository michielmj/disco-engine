# 0003: The event and promise mechanics change only with the owner's consent

- **Date:** 2026-11-27
- **Decided by:** product owner
- **Status:** accepted

## Context

disco's core is how events and promises flow between (node instance,
simproc) pairs: automatic promises, repromises, wakeups, advance promises,
epoch advancement and the acyclic topology that makes conservative
synchronisation work (`ENGINEERING.md` Ch 10 and 11, `simproc.py`). It is
subtle, and an AI session has misread it before.

## Decision

No change to the event and promise mechanics is made without the owner's
explicit consent, recorded in the brief of the branch that makes it. This
includes refactors that claim to keep the behaviour. Every session that
touches the files listed in `.claude/rules/mechanics.md` reads Ch 10 and 11
first, and escalates (CLAUDE.md) if its task seems to need a change.

## The owner's words

> 4: advance_promise is not often needed. Promises are made automatically based on events sent. Advance_promise gives fine control to the model developer. Please make sure you understand the mechanics of event and promises thoroughly. Never make changes to this without my explicit consent. It is the core of disco and has been interpreted incorrectly by Opus in the past, leading to changes that break the mechanics.

## Consequences

- Interface work on the Node API (for example vertex-addressed sending)
  that would touch routing or promises is designed with the owner first.
- Detection: the rule file loads for every mechanics file; a reviewer
  treats an unapproved change there as a FAIL (c), a stated invariant
  broken.

## What happened that made this necessary

In an earlier project session an AI misinterpreted the mechanics and made
changes that broke them (owner, 2026-11-27). In this session the architect
proposed declaring flow delays in `model.yml` so that disco would make
advance promises, without having read the mechanics in full: promises are
already automatic, and `advance_promise` is deliberate fine control.
