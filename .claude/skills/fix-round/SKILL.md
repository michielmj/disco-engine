---
name: fix-round
description: Append a fix round to a branch's brief, commit it, and resume the implementer. Use when a review reports a FAIL.
argument-hint: <branch> <round number>
---

Fix round for:

<arguments>
$ARGUMENTS
</arguments>

1. **Append `## Round N (<date>): <title>`** to the brief in the branch's
   worktree. For each finding:
   - quote it with its evidence in a `<finding>` block, so that it reads as
     data rather than as a new instruction;
   - give the remedy: *prescribed* if the reviewer gave it or you are certain
     of it, *described* if the implementer has to design it;
   - say whether a re-review follows.
2. **Commit and push the round before the session resumes.** An instruction
   that exists only in a message is invisible to `git log`.
3. **Resume the same implementer** with SendMessage: "Pull, read Round N of
   your brief, fix it, run the checks, push, and record your decisions in a
   new empty commit, with fresh Evidence lines taken on the fixed head. Your
   budget is 45 minutes, until <local time>." The earlier Evidence lines
   describe code that no longer exists; a re-review relies on the new ones.
4. **When it reports,** check for strays as /implement describes. Then
   **re-review** (/review) only if the fix changed
   behaviour, touched a security or authorization boundary, or resolved a
   finding the reviewer *described* rather than *prescribed*. Otherwise
   record "no re-review: prescribed and mutation-verified" in `STATUS.md`,
   and run /land.
