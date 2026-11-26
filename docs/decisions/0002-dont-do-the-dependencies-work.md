# 0002: Don't do the dependencies' work

- **Date:** 2026-11-26
- **Decided by:** product owner
- **Status:** accepted

## Context

disco depends on a few well-established libraries (pandas, numpy,
python-graphblas, SQLAlchemy, grpc) because they are optimised for the two
things disco is about: stability and performance. Code in disco that does
what one of them already does is code that is slower, less tested, and
coupled to the library's internals, so that each upgrade can break it.

## Decision

- disco calls its dependencies; it does not re-implement, re-derive or work
  around their behaviour. No Python loop over rows, columns, vertices or
  edges where the library has a vectorised operation.
- A deprecation or a behaviour change in a dependency is resolved by the
  dependency's documented migration, found first in its release notes,
  deprecation documentation and issue tracker, and cited in the brief.
- Where the documented way does not fit disco's contract, that is a question
  about disco's interface for the owner, not a workaround to write.
- Before any branch starts, the architect states in its brief whose job the
  behaviour is, which upstream guidance was read, and whether the change is
  on a data path; a change on a data path brings a before-and-after
  measurement in its Evidence.
- Working around or re-implementing a dependency, and a measured slowdown on
  a data path, are escalated to the owner (`CLAUDE.md`), and a review fails a
  branch that does either without the owner's approval (`REVIEW.md` (e)).

The rules for code are in `.claude/rules/dependencies.md`.

## The owner's words

> Disco is about stability and performance. We pick well established
> dependencies precisely because they are optimised on those two points. Us
> doing stuff that belongs to these dependencies introduces risk and
> performance penalties. We need to change the process to prevent this.

And, on the edit below: "We are implementing a Pyhton loop for iteration over
data, and we are replacing functionality of a well established package like
Pandas. This should not have passed review and you should have caught it as
well. Why did we not investigate what is written on the fora of Pandas first?
Did we make the assessment that this might impact performance? Did we assess
where the responsibility for this function should be?"

## Consequences

- A fix for a dependency's deprecation can take longer: reading upstream
  first, and sometimes a question to the owner, where a local shim would
  have been quick.
- Some library behaviour reaches model authors unchanged, or changes with
  the library; where that matters to disco's contract, the owner decides it
  in the interface, not in a shim.
- Detection: the brief's "Responsibility and cost" section (architect), the
  rule file (implementer), FAIL (e) and checklist item 8 in `REVIEW.md`
  (reviewer), and the architect's own reading of the production diff before
  landing (`/land`).

## What happened that made this necessary

Carry-over 11 reported that `df.fillna(default_fill)` in `graph/extract.py`
emits pandas' `Pandas4Warning` when the fill's type does not match a
column's. The architect briefed it as "no warning, and the same frames as
today". The implementer wrote `_fill_missing`: a Python loop over columns
with a hand-written table of which fill suits which dtype, and a datetime
parser, to reproduce pandas 3's results without the warning. Three review
rounds checked it only against staging's output, as the brief asked, and
found five mismatches with pandas, each fixed by more hand-written rules;
four more gaps were left as carry-overs. Nobody read pandas' guidance on
the deprecation, measured the cost, or asked whose job type coercion is.
Measured afterwards: 1.2-1.4x slower than `fillna` on a 1M-row frame, 4-5x
on a frame with nothing missing, and 4-6x across 2,000 small per-node
frames. It landed in `staging` (#54) and reached release PR #56 before the
owner caught it.
