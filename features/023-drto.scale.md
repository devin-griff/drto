# drto.scale

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that assigns scaling factors to a
declared model from its current values, and a solve that applies them,
so that models whose variables span many decades solve without
hand-written per-model factors.

```python
import drto

# ... declared, discretized, initialized model m ...

drto.scale(m)
results = drto.scaled_solve(m, solver="pounce", tee=True)
```

`drto.scale(m)` fills the standard `scaling_factor` Suffix on the model
(direction EXPORT), replacing any existing one. It reads the model's
current values, so it runs after an initializer has filled the model;
a model without values is a descriptive error saying to initialize
first.

Variable factors are assigned one per Var per name: the members of each
Var are grouped by their string index elements (species, phases, port
names), so numeric index elements (time points, spatial nodes, ordinal
counters) never split a factor. Each group's magnitude is its largest
absolute value, and a group whose magnitude falls outside [1e-2, 1e2]
gets the power of ten bringing that magnitude to order one, the exponent
clamped to twelve decades. A group whose largest value is below 1e-16
holds numerical zeros and keeps factor one. Fixed variables get no
entries: they are not part of the handed problem.

Constraint factors are measured: the Jacobian is evaluated at the
current point (PyNumero), and each constraint whose largest entry in the
scaled variables falls outside [1e-2, 1e2] gets the power of ten
bringing that entry to order one.

`drto.scaled_solve(m, solver=..., tee=..., options=...)` applies the
factors and solves. For a solver whose interface passes the Suffix
through and honors user scaling (ipopt, pounce), it solves the model
directly with `nlp_scaling_method=user-scaling`: the solver works in
scaled space internally and returns the solution, duals included, in the
model's own units, with no second model. For any other solver it falls
back to Pyomo's `core.scale_model`: build the scaled clone, solve it,
propagate the solution back. The two paths solve the same scaled
problem; the fallback costs the clone and maps only the primal solution
back.

The factors compose with the rest of the package: the initializers
already honor an active `scaling_factor` Suffix on their internal solves
(feature 011), so the order is initialize, scale, solve.

## Benefit hypothesis

Models taken from flowsheet libraries carry their physics' units: the
solvent extraction stage holds concentrations across twelve decades and
a Jacobian spanning seventeen, and its dynamic optimization fails from
its own steady state — the KKT factorization is numerically singular.
With factors from this rule and the measured constraint factors, the
same solve converges in eight iterations. One call after initialization
replaces the per-model scaling work that today only the CSTR example's
hand-written units-driven factors demonstrate.

## Acceptance criteria

- `drto.scale(m)` fills the `scaling_factor` Suffix with powers of ten
  and replaces any existing Suffix on repeat calls. Members of one Var
  that differ only in numeric index elements share a factor; members
  that differ in a string index element may not.
- A group whose magnitude lies inside [1e-2, 1e2], and a group whose
  largest value is below 1e-16, get no entry. Fixed variables get no
  entries. The exponent clamp holds at twelve decades.
- After the call, every constraint's largest Jacobian entry in the
  scaled variables lies inside [1e-2, 1e2] at the measured point, for
  every constraint whose unscaled row was outside.
- A model without values raises a descriptive error saying to
  initialize first.
- `drto.scaled_solve` with ipopt or pounce solves the model directly
  under `nlp_scaling_method=user-scaling`, no clone built, and returns
  the solution in the model's own units. With a solver that does not
  honor the Suffix it builds the `core.scale_model` clone, solves it,
  and propagates the solution back. Both paths reach the same solution
  on a test model.
- The solvent extraction example's dynamic optimization converges from
  its steady start through `drto.scale` and `drto.scaled_solve`.
- The initializers run their internal solves against the assigned
  factors unchanged (feature 011's scaled-clone behavior), with no
  duplicate assignment.
