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

## The declarations

The declaration surface (feature 002) tags the pieces of an optimization or
simulation problem on a model you already built: `declare_time`,
`declare_state`, `declare_continuous_dynamics`, `declare_control` (with its
pyomo-cvp `profile`), the stage and terminal costs, `declare_initial_condition`
(a state at the first time point equal to a mutable Param, the feedback hook),
`declare_terminal_constraint`, and the steady-state targets. Each declaration
validates its convention and records the component in the registry, where the
transformations find it. Declarations that scale with the states and controls
take varargs and accumulate; the one-of-each declarations error on a second,
different object. Conventions are read from either side of the written
equality, so `lhs == rhs` and `rhs == lhs` are equivalent.
