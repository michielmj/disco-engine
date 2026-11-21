---
name: prove-a-rule
description: How to pin a requirement rule with a test that provably fails when the rule is broken (mutation verification). Use when implementing any rule from a brief, or when a review asks whether a test enforces its rule.
---

# Pinning a rule

For each rule your brief names:
1. Write the test against the property itself (`.claude/rules/testing.md`).
2. Break the rule in the source: drop the check, widen the condition, return
   early, or swap the value it depends on.
3. Run the named test, and require that test, not some other one, to fail.
   Run only the test files the mutation concerns, never the full suite once
   per mutation: the machine is shared.
4. Restore the source, and run the test until it passes.
5. Record the rule, the test, the mutation and the failure in your report,
   and in the Evidence line of your Surfaced decisions commit
   (record-decisions), so a reviewer need not run it again.

<example>
Rule 4: an export of exactly the size limit is accepted.
Test: `tests/test_export.py::test_export_at_the_limit_is_accepted` exports
`limit` rows, and asserts the status is 200 and the file was written.
Mutation: `rows > limit` became `rows >= limit` in `src/disco_data/export.py:88`.
Failure: `AssertionError: 413 != 200` in that test; the rest of the suite
stayed green.
</example>

A test that survives every mutation is a tautology, and a FAIL in review.
Typical causes:
- an assertion on a constant, or on a value the test computed itself;
- an assertion on a mock's call count;
- a fixture that cannot reach the code path.

A test named after a defect must fail when that defect's fix is reverted.
