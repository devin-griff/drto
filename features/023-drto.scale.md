# drto.scale

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want a function that assigns scaling factors to a
declared model from a source I choose, and a solve that applies them,
so that models whose variables span many orders of magnitude solve
without hand-written per-model factors.

```python
import drto
from pyomo.environ import units

# ... declared, discretized, initialized model m ...

drto.scale(m)                       # magnitudes from the values m holds
drto.scale(m, source="bounds")      # from the declared bounds
drto.scale(m, source={units.J: 1e7, units.W: 1e6})  # from the caller
results = drto.scaled_solve(m)
results = drto.scaled_solve(m, source="bounds", solver="ipopt_v2",
                            tee=True, options={"max_iter": 500})
```

`drto.scale(m, source=...)` fills the standard `scaling_factor` Suffix
on the model (direction EXPORT), replacing any existing one. `source`
selects where each variable group's magnitude comes from, and the modes
are exclusive: each scales the groups it can measure and leaves every
other group at factor one. Whatever the mode, the constraint factors
are measured through the Jacobian at the model's current point, so
`scale` runs after an initializer has filled the model, and a model
without values is a descriptive error saying to initialize first.

Variable factors are assigned one per Var per name: the members of each
Var are grouped by their string index elements (species, phases, port
names), so numeric index elements (time points, spatial nodes, ordinal
counters) never split a factor.

The mode gives each group its magnitude. `"point"`, the default, reads
the group's largest absolute value at the point the model is sitting
at. `"bounds"` reads `max(|lb|, |ub|)` over the members carrying two
finite bounds, and a group with no such member keeps factor one, so the
mode scales exactly the quantities whose operating limits are declared,
in practice the controls, and reads no values. A mapping of units to
magnitudes, `{units.J: 1e7, units.W: 1e6}`, gives every group whose
members are valued in a mapped dimension that dimension's magnitude,
and every other group keeps factor one: the caller states the process's
operating magnitudes once per physical dimension, which covers a
quantity whose value and bounds both say nothing, a duty sitting at
zero with no bounds. There is no mode named `"units"`, the mapping
itself is the mode, so a units request without magnitudes cannot be
written. Any other `source` is a ValueError naming the three forms.

Whatever the mode, a group whose magnitude falls outside [1e-2, 1e2]
gets the power of ten bringing that magnitude to order one, the
exponent clamped to twelve orders of magnitude. A group whose magnitude
is below 1e-16 holds numerical zeros and keeps factor one. Fixed
variables get no entries: they are not part of the handed problem.

A derivative variable on the terminal segment takes the factor of the
state it differentiates, reached through `get_state_var` and the pairing
`infinite_horizon` records, rather than one measured from its own value.
The segment's derivatives go to zero at the equilibrium the tail
approaches, so the magnitude measured there is a zero rather than a
scale, and measuring one gives the group a factor as large as the clamp
allows.

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

The Jacobian is evaluated through PyNumero, which needs its `pynumero_ASL`
library findable. `pyomo download-extensions` installs it where pyomo
looks; IDAES ships the same library and registers its directory when
`idaes` is imported, which is why an IDAES model scales with nothing
further done. drto imports neither: when the library is absent, `scale`
raises an error naming both ways to install it, rather than letting
PyNumero fail on a null handle.

The infinite horizon endpoint pins are skipped completely: the pin
slacks get no variable factors and the pin constraints get no
constraint factors. The pin's penalty weight is defined in each state's
own units, and a factor on the slack or its constraint changes the
pin's effective weight against the objective.

`drto.scaled_solve(m, source=..., solver=..., tee=..., options=...)`
assigns the factors and solves. `source` is forwarded to `drto.scale`,
so every call measures fresh and replaces a Suffix already on the
model. `solver` names the solver and defaults to `pounce_v2`; `options`
is a mapping passed through to it, overriding any default the call
sets.

The factors reach the solver through the Suffix, and no second model is
built anywhere. pounce and legacy ipopt read it under
`nlp_scaling_method=user-scaling`: objective and constraint factors
travel through the NL file's suffix segments, and variable factors are
applied as a change of variables inside the solver. ipopt_v2 gets no
scaling option, since Pyomo's NL-v2 writer consumes the Suffix and
scales the problem as it writes the file, so that solver receives an
already-scaled problem. Every route returns the solution in the model's
own units.

A solver outside that set does not receive the factors: the solve runs
unscaled, and `scaled_solve` warns, naming the solver and saying the
factors were not applied.

The factors compose with the rest of the package: the initializers run
in the model's own units (features 010 and 011), and the loops honor an
active Suffix or write one through their `scale` option (features 014
and 026), so the order is initialize, scale, solve.

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
  magnitude is below 1e-16, get no entry. Fixed variables get no
  entries. The exponent clamp holds at twelve orders of magnitude.
- With `source="bounds"`, a group with a member carrying two finite
  bounds takes its factor from the largest absolute bound and no value
  is read: a control sitting at zero with bounds at plus and minus 1e6
  takes 1e-6. A group with no such member keeps factor one.
- With `source` a mapping of units to magnitudes, a group valued in a
  mapped dimension takes that magnitude's factor, and every other
  group, unmapped or dimensionless, keeps factor one. The mapping is
  the only units form: no string selects a units mode.
- A `source` that is none of `"point"`, `"bounds"`, or a mapping raises
  a ValueError naming the three forms.
- `drto.scaled_solve` forwards `source` to `drto.scale`, and each call
  replaces the Suffix a previous call wrote.
- After the call, every constraint whose largest Jacobian entry in the
  scaled variables exceeded 1e2 has been brought to order one, and a
  constraint whose entries were all small carries no factor.
- A model whose objective is the constant zero scales like any other,
  and a model with no objective raises no error.
- Without PyNumero's `pynumero_ASL` library, `scale` raises a descriptive
  error naming `pyomo download-extensions` and IDAES's extensions. drto
  imports neither pyomo's extension machinery nor IDAES to find it.
- A model without values raises a descriptive error saying to
  initialize first.
- On a model carrying a terminal segment, each of the segment's
  derivative members carries the factor of the state member it
  differentiates, rather than one measured from its own value, so a
  segment sitting at its equilibrium does not drive those groups to the
  clamp.
- On a model carrying a terminal segment, the endpoint pin slacks and
  the pin constraints have no entries in the Suffix, so the pin's
  effective weight against the objective is the declared one, unchanged
  by scaling.
- `drto.scaled_solve` takes the solver by name, defaulting to
  `pounce_v2`, and passes an options mapping through to it.
- With pounce or legacy ipopt it solves the model under
  `nlp_scaling_method=user-scaling`; with ipopt_v2 it passes no scaling
  option, since the NL-v2 writer applies the factors as it writes. No
  clone is built on any route, and the solution comes back in the
  model's own units.
- With a solver that does not receive the factors, the solve runs
  unscaled and a warning names the solver and says the factors were not
  applied.
- The steady reduction of the solvent extraction example converges
  through `drto.scale` and `drto.scaled_solve`, with the concentrations
  and flows carrying their bounds at zero: the handed point survives
  the solver's move off those bounds.
