---
name: review
description: Launch a read-only review of a task branch. Use when an implementer reports a branch done, or after a fix round that needs re-review.
argument-hint: <branch> <head sha>
---

Review this branch:

<arguments>
$ARGUMENTS
</arguments>

1. **Create the reviewer's worktree:** `bash scripts/new_task.sh --review <branch>`.
2. **Write the `<likely_wrong>` list:** five to eight numbered items,
   specific to this diff, each one the reviewer can confirm or refute. A
   focused reviewer finds defects; an unfocused one finds spellings. Draw the
   items from:
   - the implementer's report: what it flagged, decided, or could not check;
   - every invariant the diff touches;
   - security boundaries;
   - anything a user sees.

   Take items from the report, but never paste the report into the prompt:
   a reviewer who reads the implementer's account checks the account, not
   the code.

   Don't ask for a full-suite re-run or extra mutations: the reviewer relies
   on the Evidence lines of the Surfaced decisions commit (REVIEW.md §
   Proportion), unless they are missing.
3. **Spawn the reviewer** in the background, with this prompt, filled in:

   ```
   <task>
   branch: <branch>
   head: <sha>
   worktree: <absolute path of the review worktree>
   ports: <ports>
   word limit: 900
   budget: 45 minutes (20 for a scoped re-review), until <local time>
   report file: <absolute path of .claude/worktrees/review-<branch>.md>   (beside the worktree, not in it)
   </task>
   <likely_wrong>
   1. <item>
   </likely_wrong>
   Review the branch, as your agent definition and REVIEW.md describe.
   ```
4. **When it reports,** check for strays as /implement describes. If its
   reply never arrives, read the report file: a hand-back can fail. Then
   remove its worktree (`git worktree remove <path>`) and the report file,
   and:
   - **0 FAIL:** run /land. Fold any observation that concerns the next task
     into that task's brief, and put the rest in `STATUS.md`'s backlog.
   - **Any FAIL:** run /fix-round.
