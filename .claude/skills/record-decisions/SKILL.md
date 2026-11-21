---
name: record-decisions
description: How to write the Surfaced decisions block that ends every session on a branch, with an example. Use before your final commit on a branch, and when correcting an earlier block.
---

# Recording your decisions

End the session with an empty commit whose message holds the block:

```bash
git -c user.name="<INITIALS> (<Role> Agent)" commit --allow-empty -F - <<'EOF'
Record this session's Surfaced decisions

## Surfaced decisions
- ASSUMED: ...
EOF
```

Each entry is a sentence or two, with the evidence a reviewer needs to check
it. Write "none" under a heading rather than leaving the heading out.
- **ASSUMED:** what you read into an instruction that could be read two
  ways, and why you chose that reading.
- **DECIDED:** a judgement call within your remit, and the alternative you
  rejected.
- **ESCALATED:** what you stopped on, and whether it blocks the branch.
- **DISCREPANCY:** a statement in the brief or the docs that did not hold,
  with the evidence.
- **Evidence** (implementers; the proportionate-effort decision): the test
  counts of your last full run and the head it ran on, each mutation and the
  test it failed, and what the "somebody has used it" steps showed. A reviewer relies on these
  lines instead of re-running everything, so record them once, here.

<example>
## Surfaced decisions
- ASSUMED: "Large exports" in rule 3 means over the configured limit, not
  over 1 GB: the brief's done-when steps use the setting by name.
- DECIDED: Streamed the export through a generator instead of a temporary
  file, so memory stays flat (tests/test_export.py::test_flat_memory); a
  temporary file would need a clean-up path on every error.
- ESCALATED: none.
- DISCREPANCY: The brief says `export_rows` returns a list; it returns a
  generator (src/disco_data/export.py:41). I built on the generator.
- Evidence: full suite 412 passed, slow 9 passed, at 3f2c1ab; mutation:
  `>` changed to `>=` in export.py:88 fails
  tests/test_export.py::test_export_at_the_limit_is_accepted; used: the
  `export` command on the shipped sample wrote 1,204 rows.
</example>

The reviewer checks the block against the diff. A decision made silently is a
finding.

## Correcting a block

Never amend. Add a second empty commit, `Correct the Surfaced decisions in
<sha>`, that names the false claim and says whether its conclusion survives.
