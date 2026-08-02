# Design

drto is developed spec-first: every capability is written down as a
short specification before it is implemented. A spec states the user
story, the benefit hypothesis the capability is betting on, and testable
acceptance criteria; the criteria drive the tests, and the
implementation is done when they pass. Specs are amended as the design
sharpens, so they remain the authoritative statement of what each piece
does and why it is shaped the way it is. When these docs and a spec
disagree, the spec wins and the docs have a bug.

The specs live in the repository under
[`features/`](https://github.com/devin-griff/drto/tree/main/features),
and the
[design record](https://github.com/devin-griff/drto/blob/main/DESIGN.md)
holds the cross-cutting decisions. A feature request is a pull request
adding a spec file under `features/` in the template format.
