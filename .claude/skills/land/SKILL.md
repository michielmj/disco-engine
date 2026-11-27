---
name: land
description: Land a reviewed task branch into staging with a merge commit. Use after a review with no FAIL, after a fix round that needs no re-review, or for a prescribed small change once you have checked its diff.
argument-hint: <branch>
---

Land this branch:

<arguments>
$ARGUMENTS
</arguments>

1. **Fetch:** `git fetch origin`. Your own working tree must be clean. A
   branch stacked on another lands only after its parent, by its own PR into
   `staging`: never into the parent. If its PR targets the parent,
   retarget it to `staging` once the parent has landed.
2. **Read the production diff yourself,** every changed file under `src/`,
   not the reports about it: `git diff origin/staging...origin/<branch> -- src`.
   Ask of each change whose job it is and what it costs (decision 0002,
   `REVIEW.md` checklist item 8). A change that does a dependency's work or
   slows a data path without the owner's approval does not land: run
   /fix-round, or bring the question to the owner.
3. **Trial merge:** `git merge-tree --write-tree --name-only origin/staging origin/<branch>`.
   - **A conflict:** resume the implementer to merge `origin/staging` into
     its branch, keeping both sides of the docs. A merge that changes
     behaviour gets a scoped look from a reviewer.
   - **Both sides changed a generated file:** regenerate it on a detached
     worktree of the trial merge, and check that nothing drifted. Never trust
     a textual merge of a generated file.
4. **Open a PR** from the branch to `staging`. Its description gives what
   was built, by rule; the review's verdict, or "prescribed small change, no
   review" with what you checked; the fix rounds; the checks; and what is
   left for the owner.
   - GitHub: `gh pr create --head <branch> --base staging --title "<title>" --body-file <file>`
5. **Complete it with a merge commit,** never a squash: the empty commits
   hold the reasoning. Keep the source branch.
   - GitHub: `gh pr merge <number> --merge` (without `--delete-branch`)
6. **Confirm** that `origin/staging` has the merge commit. Remove the task's
   worktree (`git worktree remove .claude/worktrees/<branch>`), which frees
   its ports, and update `STATUS.md`.
