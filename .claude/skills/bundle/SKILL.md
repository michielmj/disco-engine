---
name: bundle
description: Pre-check staging and open the release-bundle PR to main for the owner's vote. Started by the owner.
argument-hint: <target version, e.g. v0.4.0-rc1>
disable-model-invocation: true
---

Open the release bundle for:

<arguments>
$ARGUMENTS
</arguments>

1. **Confirm with the owner** that they want this bundle now. A release is
   theirs to schedule, and every touch of `main` costs a full pipeline run.
2. **Pre-check `staging`,** on a local branch at `origin/staging`:
   - `bash scripts/test_linux.sh <that branch>`, and again with `-- -m slow`;
   - a trial merge onto `main`;
   - a check that `main`'s pipeline file is not overwritten by an older one.

   When a slow test fails, rerun it alone and loop it to capture the
   message, and run an A/B against `main` before you call it a regression.
   Record what you find in `STATUS.md`'s backlog.
3. **Open a PR from `staging` to `main`.** Its description gives:
   - the contents since the last release, by milestone;
   - the upgrade notes: format versions, migrations, operator steps;
   - the pre-check results;
   - the reminder "keep the source branch: do not delete `staging`";
   - what only the owner can do: vote, merge, tag, deploy, and anything that
     needs production or a real external service.
4. **Stop there.** The vote, the merge and the tag are the owner's.
