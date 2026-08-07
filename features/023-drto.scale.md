# drto.scale

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that assigns scaling factors to a
declared model from its current values, and a solve that applies them,
so that models whose variables span many orders of magnitude solve
without hand-written per-model factors.

```python
import drto

# ... declared, discretized, initialized model m ...

drto.scale(m)
results = drto.scaled_solve(m)
results = drto.scaled_solve(m, solver="ipopt", tee=True,
                            options={"max_iter": 500})
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
clamped to twelve orders of magnitude. A group whose largest value is
below 1e-16 holds numerical zeros and keeps factor one. Fixed variables
get no entries: they are not part of the handed problem.

The floor sits at machine zero rather than at the precision of whatever
filled the model. A higher floor leaves trace quantities unscaled, and
an unscaled variable's bound is what the solver's starting point then
displaces it away from: the solvent extraction stage carries chloride at
1e-7 mg/L against a bound at zero, and moving it to the interior by a
fixed absolute amount changes it by five orders of magnitude.

Constraint factors are measured: the Jacobian is evaluated at the
current point (PyNumero), and each constraint whose largest entry in the
scaled variables exceeds 1e2 gets the power of ten bringing that entry
to order one. Rows whose entries are all small are left alone. Scaling
a row up multiplies its residual along with its entries, so a row whose
terms cancel to their rounding floor becomes a violation: the same
stage's chloride initial condition holds terms near 1e-9 and a residual
of 2e-10, which a factor of 1e9 turns into a violation of 0.2.

A model whose objective is the constant zero, which is what
`drto.steady_state_simulation` installs, scales like any other. A model
carrying no objective at all is handled rather than rejected, since the
Jacobian measurement reads the objective through PyomoNLP.

The infinite horizon endpoint pins are skipped completely: the pin
slacks get no variable factors and the pin constraints get no
constraint factors. The pin's penalty weight is defined in each state's
own units, and a factor on the slack or its constraint changes the
pin's effective weight against the objective.

`drto.scaled_solve(m, solver=..., tee=..., options=...)` applies the
factors and solves. `solver` names the solver and defaults to
`ipopt_v2`; `options` is a mapping passed through to it, overriding any
default the call sets. For a solver whose interface passes the Suffix
through and honors user scaling, it solves the model directly with
`nlp_scaling_method=user-scaling`: the solver works in scaled space
internally and returns the solution, duals included, in the model's own
units, with no second model. ipopt does this today; pounce's algorithm
honors user scaling but `pyomo_pounce` does not yet pass the Suffix to
it, so pounce takes the fallback until that lands. The fallback is
Pyomo's `core.scale_model`: build the scaled clone, solve it, propagate
the solution back. The two paths solve the same scaled problem; the
fallback costs the clone and maps only the primal solution back.

The factors compose with the rest of the package: the initializers
already honor an active `scaling_factor` Suffix on their internal solves
(feature 011), so the order is initialize, scale, solve.

## Benefit hypothesis

Models taken from flowsheet libraries carry their physics' units: the
solvent extraction stage holds concentrations across twelve orders of
magnitude and a Jacobian spanning seventeen, and its dynamic
optimization fails from its own steady state.

The bounds are where this bites hardest. An interior-point method moves
every variable that sits near a bound away from it before the first
step, by a fixed amount measured in the units the solver sees. Scaled,
that displaces each variable relative to its own size. Unscaled, it
displaces the stage's trace species from 1e-7 to 1e-2, which no longer
satisfies the equilibria or the dissociation, so the solver starts from
a broken point however good the one it was handed. With factors
assigned, the same model with the same bounds starts primal-feasible at
9e-9 and converges in four iterations.

The factors also carry the setpoint solve. The steady reduction of that
stage is solved through the same rule, which is what gives the trace
species values worth scaling in the first place.

One call after initialization replaces the per-model scaling work that
today only the CSTR example's hand-written units-driven factors
demonstrate.

## Acceptance criteria

- `drto.scale(m)` fills the `scaling_factor` Suffix with powers of ten
  and replaces any existing Suffix on repeat calls. Members of one Var
  that differ only in numeric index elements share a factor; members
  that differ in a string index element may not.
- A group whose magnitude lies inside [1e-2, 1e2], and a group whose
  largest value is below 1e-16, get no entry. Fixed variables get no
  entries. The exponent clamp holds at twelve orders of magnitude.
- After the call, every constraint whose largest Jacobian entry in the
  scaled variables exceeded 1e2 has been brought to order one, and a
  constraint whose entries were all small carries no factor.
- A model whose objective is the constant zero scales like any other,
  and a model with no objective raises no error.
- A model without values raises a descriptive error saying to
  initialize first.
- On a model carrying a terminal segment, the endpoint pin slacks and
  the pin constraints have no entries in the Suffix, so the pin's
  effective weight against the objective is the declared one, unchanged
  by scaling.
- `drto.scaled_solve` takes the solver by name, defaulting to
  `ipopt_v2`, and passes an options mapping through to it.
- With a solver whose interface passes the Suffix through it solves the
  model directly under `nlp_scaling_method=user-scaling`, no clone
  built, and returns the solution in the model's own units. With one
  that does not it builds the `core.scale_model` clone, solves it, and
  propagates the solution back. Both paths reach the same solution on a
  test model.
- The solvent extraction example's dynamic optimization converges from
  its steady start through `drto.scale` and `drto.scaled_solve`, with
  the concentrations and flows carrying their bounds at zero: the
  handed point survives the solver's move off those bounds.
- The initializers run their internal solves against the assigned
  factors unchanged (feature 011's scaled-clone behavior), with no
  duplicate assignment.
