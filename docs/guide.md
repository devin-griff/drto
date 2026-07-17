# User guide

The narrative guide grows as the core lands: the six modes, the declaration
surface, and the initialization routines. Until then, the
[README](https://github.com/devin-griff/drto#readme) is the overview and
`DESIGN.md` is the authoritative design record.

## The registry: `drto.info`

Every drto model carries one registry: the record of what has been declared
and which transformations have been applied. `drto.info(m)` returns it,
creating it on first access. It lives in Pyomo's namespaced private data, so
it never appears in the model's component tree, and it survives `clone()`
(and every transformation's `create_using` form) with its stored component
references remapped to the clone.

Displaying it renders a drto-aware view of the model: declarations grouped by
role, indexed constraints as one symbolic equation per family (for example
`dzdt[k] == - z[k] + u[k]  for k in t`, not the per-index expansion `pprint`
produces), and the ordered log of applied transformations with their
outcomes.

The declarations (feature 002) write to the registry and the transformations
read it, so it is the one place drto looks for the model's declared pieces.
