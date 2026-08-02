# Spec-first Development

drto is developed spec-first: every capability is written down as a
short specification before it is implemented. Specs are amended as the
design sharpens, so they remain the authoritative statement of what
each piece does and why it is shaped the way it is. When these docs and
a spec disagree, the spec wins and the docs have a bug.

Every spec has the same three parts:

- **Description**: the user story. What the capability is, the call
  that invokes it, and what happens when it runs, written concretely
  enough that the reader can picture the session before any code
  exists.
- **Benefit hypothesis**: the bet. Why the capability is worth
  building, stated as the benefit it should deliver, so the value of
  what shipped can be judged against what was promised.
- **Acceptance criteria**: the contract. Testable statements of
  behavior that drive the tests; the implementation is done when every
  criterion passes.

The specs live in the repository under
[`features/`](https://github.com/devin-griff/drto/tree/main/features).
A feature request is a pull request adding a spec file under `features/`
in the template format.
