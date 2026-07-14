# dev-notes

drto's durable engineering memory: design records, code-review logs, progress
trackers, and forward-looking research. This is the one place development
history is allowed to accrete, which keeps code comments clean (they state
present-tense rationale only).

`DESIGN.md` at the repo root stays the top-level design record. These notes
are the working layer beneath it.

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
