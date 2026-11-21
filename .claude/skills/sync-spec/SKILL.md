---
name: sync-spec
description: Bring ENGINEERING.md in line with a branch's changes - find the chapters the diff affects and update only those sections. Use before the Surfaced decisions commit of any branch that changed a contract, and when a review finds the spec out of date.
---

# Keeping ENGINEERING.md true

`ENGINEERING.md` is the specification: contracts and invariants, by chapter
(the list is in CLAUDE.md). A branch that changes a contract updates the spec
in the same branch.

1. List what the branch changed: `git diff --stat origin/staging...HEAD`.
2. Map each changed module to its chapter, and read that chapter in full
   before you edit it.
3. Update only the sections the change touched, at the spec's level: what is
   guaranteed, to whom, and under which conditions. Leave implementation
   detail (private helpers, loop structure) out.
4. Don't rewrite sections the branch did not touch, even when you would
   phrase them differently: an unrelated edit hides the real change from the
   reviewer.
5. If the spec said something the code never did, that is a DISCREPANCY:
   record it in the Surfaced decisions, and escalate if the code, not the
   spec, looks wrong.
