---
name: implement
description: Cut a task branch, commit its brief, and launch an implementer on it. Use when an approved requirement has a branch ready to build.
argument-hint: <branch> <requirement file> <rules, e.g. "rules 1-6">
---

Start this task:

<arguments>
$ARGUMENTS
</arguments>

0. **Size it** (the proportionate-effort decision). A process sized for the
   large case taxes every small one.
   - **A prescribed small change** (the owner gives the code, up to about 20
     lines): brief and implementer as below, then check the diff yourself
     against the prescription and its Evidence lines. No reviewer, unless it
     touches security, personal data or data loss. Land it with /land.
   - **Follow-ups to the owner's own code:** one branch for the whole review
     of that code, each later point a fix round on it, not a new branch per
     point.
   - **Everything else:** as below, with one review.
1. **Create the worktree:** `bash scripts/new_task.sh <branch>`. For a task
   stacked on an unmerged parent, give the parent as the base:
   `bash scripts/new_task.sh <branch> origin/<parent>`. Note the worktree
   path and the ports it prints.
2. **Write the brief,** `specs/briefs/<branch>.md` in that worktree, from
   `specs/briefs/_TEMPLATE.md`:
   - the goal: the requirement's rules, by number. Point to them; never copy
     them, or the copy drifts.
   - responsibility and cost (decision 0002), filled in before anything
     else: whose job each behaviour is; for a dependency's behaviour, its
     release notes, deprecation docs and issue tracker, read by you, linked,
     with what they recommend; and whether the change is on a data path,
     with the benchmark it must bring. If the right fix belongs to the
     dependency, or changes what a user sees, the brief waits for the
     owner's answer. Never brief "same result as today" for behaviour a
     dependency is deprecating.
   - where this is most likely to go wrong. This is where your judgement
     adds most, so spend it here, and write each item as an acceptance test.
     Keep to main paths and to rules on security, personal data or data
     loss: an edge case without a reason is not a test to ask for
     (`.claude/rules/testing.md`).
   - no migration or PR numbers that don't exist yet: "the next free
     number".
   - the constraints, and the definition of done, including "somebody has
     used it".
   - facts about the code: only those you have verified. Leave the rest of
     the survey to the implementer.
3. **Commit the brief** as the branch's first commit, so the reviewer sees
   the instruction and the result in one diff. Push it:
   `git -C <worktree> push -u origin <branch>`.
4. **Spawn the implementer** in the background, with this prompt, filled in:

   ```
   <task>
   branch: <branch>
   worktree: <absolute worktree path>
   brief: specs/briefs/<branch>.md
   ports: <ports>
   run settings: <what the app needs to run on these ports, such as its public URL, or "none">
   budget: 90 minutes, until <local time>
   </task>
   <parallel_sessions>
   <branch>: <area of the code>, ports <ports>    (one line each, or "none")
   </parallel_sessions>
   Build the brief, as your agent definition describes.
   ```
   Budgets: implementation 90 minutes, a fix round 45, a review 45, a
   scoped re-review 20, a requirement review 30. At its budget an agent
   stops and reports what it has; you decide whether it continues.
5. **When it reports,** check for strays before anything else: `docker ps`,
   and the processes running under its worktree path. Stop what it left
   behind, by pid or container name, and say so in your next brief. Treat a
   "background work still running" notice as a stray, never as a glitch.
   Then compare the report with the brief, and run /review.
