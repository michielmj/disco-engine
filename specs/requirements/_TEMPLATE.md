# NN: <title>

**Status: draft for the product owner's approval, <date>, revision 1.**
- **Next:** a review of this draft, then the owner's answers to any
  questions it raises, then the owner's approval. Nothing is briefed until
  then.
- **Once approved,** this document wins wherever it and a brief disagree.

<!-- Each revision adds one line here: what its review found, and what
changed. Once approved, the status line becomes "approved by the product
owner, <date> ("<their words>"), revision N." -->

## The requirement, in the product owner's words

<date>:

> <Quote the owner verbatim. Do not tidy it.>

## How this was settled

**The owner's answers, verbatim.**

| Round | Question | Answer |
|---|---|---|
| 1 | <the question as asked> | "<their words>" |

**What exists today** (surveyed <date>, at `<branch>` `<sha>`):
- <The facts the rules depend on, each verified, with a file path where it
  helps.>

## Rules

<!-- Number the rules, and tag each one (owner) or (architect). Put the owner's
answers in the rules, not only in the table. Each rule must be testable: a
test can fail on it.
  Testable:     "An export of more than APP_EXPORT_LIMIT rows is refused with a
                 message that names the limit, and nothing is written."
  Not testable: "Exports are fast and safe."
A rule that can be restated in more than one way is still a question for the
owner. Name no migration or PR number that does not exist yet: "the next free
number". -->

1. **<Rule in one bold sentence>** (owner). <Detail.>
2. **<Rule>** *(architect)*. <Detail, and why.>

## Out of scope

- <What was not chosen, with the owner's words where they gave them.>

## How the work is split *(architect)*

- **NNa, <part>:** rules x–y. <Why this boundary: review, security,
  dependency.>
- **NNb, <part>:** rules … <Stacked on NNa, or cut after it lands.>

## What this changes in `CLAUDE.md`, `ENGINEERING.md` and the user guide

- <Each document, and what it gains.>
