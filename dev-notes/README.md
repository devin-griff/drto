# dev-notes

drto's durable engineering memory: design records, code-review logs, progress
trackers, and forward-looking research. This is the one place development
history is allowed to accrete, which keeps code comments clean (they state
present-tense rationale only).

The specs in `features/` are the design record. These notes are the
working layer beneath them.

## Naming so notes are findable

Use these patterns so an agent can Glob for a genre:

- `code-review-YYYY-MM.md` -- a dated, severity-ranked review, read-only once
  written; follow-ups go in `code-review-YYYY-MM-followups.md`.
- `issue-NNN-<slug>.md` -- work notes for a specific issue.
- `<feature>-progress.md` -- live state for a multi-session or loop task, with
  `[ ]` / `[x]` checkboxes.
- `release.md` -- the release runbook, once there is one.
- `research/` -- forward-looking work not tied to a single change.

## House style for a note

Open with a one-line **Status**. When a note's framing is overtaken by later
findings, append a "Progress / current state" section that corrects the
original rather than editing it away, so the correction stays visible.

## Progress trackers

A progress tracker is a `/loop`-walkable worklist. Two shapes, by what the
items are:

- **Status table** for triaging found problems (a code review, a tech-debt
  sweep): one row per item with an ID, the finding, a status (`open`,
  `fixed`, `verified-deferred`), and a receipt.
- **Checkbox list** for executing planned tasks (a feature phase, a
  migration): one `[ ]` per task, each with an acceptance check; flip to
  `[x]` with a dated one-line result when it passes.

The rule for both: nothing is marked done without an inline verification
receipt. For a fix, state how the mechanism was confirmed, the test name, and
that the test fails before the fix and passes after. For a task, state the
acceptance check that passed. `verified-deferred` records the same
confirmation but explains why the change is not being made here.

Loop protocol: one item per iteration, do exactly that item, run its check,
update its row or box, then stop. Do not batch; the value is small, verifiable
steps.

Keep a review and its tracker separate: the dated review note
(`code-review-YYYY-MM.md`) is read-only once written and status moves in the
tracker, so the findings stay stable while progress accrues.
