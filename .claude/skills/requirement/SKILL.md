---
name: requirement
description: Start a requirement - survey the code, ask the owner, draft it, and review it until it is ready to brief. Started by the owner.
argument-hint: <short topic, e.g. "export to Excel">
disable-model-invocation: true
---

Start a requirement for this topic:

<topic>
$ARGUMENTS
</topic>

1. **Survey.** Find out what exists before you draft anything: questions
   asked without a survey are vague, and answers given without one are
   guesses. If the topic spans many files, spawn the surveyor with the
   factual questions in a `<questions>` block; otherwise read the code
   yourself.
2. **Ask the owner** the product questions the survey leaves open, with
   AskUserQuestion: at most four a round, each with options and your
   recommendation. Ask only what changes what a user sees, and settle the
   rest yourself. An answer that can be read two ways is asked again, with an
   example of each reading. Put each question in the owner's terms, at one
   layer: what a user sees, not how the code does it. A question that mixes
   layers gets "I don't understand the question".
3. **Draft** `specs/requirements/NN-<slug>.md` from `_TEMPLATE.md`:
   - the owner's words verbatim, in "How this was settled";
   - numbered rules, each testable and tagged *(owner)* or *(architect)*;
   - how the work splits into branches.
4. **Commit it** on a branch `req/NN-<slug>` cut from `origin/staging`, and
   push.
5. **Check your own facts first** (the proportionate-effort decision):
   every statement in "What exists today" and every code path a rule names,
   against the code. Name no migration or PR number that does not exist
   yet: "the next free number".
   The cheapest review is the author's; the reviewer should find what you
   could not.
6. **Review it.** Spawn the requirement-reviewer with the draft's path, a
   budget of 30 minutes with its end time, and a report file to write to
   (`.claude/worktrees/req-NN-round-<n>.md`, gitignored) in case its reply
   never arrives. Put
   its questions for the owner to the owner, revise, and add a line per
   revision to the draft's status block.
   - **A small requirement** (one branch, on existing patterns): one round.
     Fix what it finds without another round, unless it raises a product
     question.
   - **A large requirement:** repeat until the reviewer says "ready to
     brief"; two to five rounds is normal.
7. **Open a PR to `staging`** for the owner to approve and complete. Nothing
   is briefed before it merges.
