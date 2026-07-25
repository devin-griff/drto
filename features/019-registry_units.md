# Registry units

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want the registry view to show the units of everything I
declared, so that a unit-carrying model (an IDAES flowsheet, or any model built
with `pyo.units`) reads back with its physics visible, and so that an equation
whose units do not balance is flagged where I am already looking.

```python
drto.info(m)
# states:   material_holdup (free, mol), energy_holdup (free, J)
# controls: heat (piecewise_constant, free, W)
# dynamics: material_accumulation[t,p,j]  ==  ...  for t in t  (mol/s)
# tracking stage cost: stage[t]  ==  ...  for t in samples  (inc)
```

Every registry row is annotated the same way, from `pyo.units.get_units`:

- A variable or parameter row reports the component's units, appended to the
  parenthetical notes it already renders (`free`, `piecewise_constant`, ...).
- A constraint row reports its body's units, a parenthetical after the
  `for ...` clause, which is what the equation balances in (`mol/s` for a
  material balance, `W` for an energy balance).
- Dimensionless renders as nothing: a model written without units displays
  exactly as it does today.
- A constraint body whose units are inconsistent renders `(inc)` in place of
  the unit. Display only; declaration does not warn or reject.

Units render compact where pint can (`J`, `W`), not as base units
(`kg*m**2/s**2`).

## Benefit hypothesis

The registry is the one place a declared model is read back as a whole, so it
is where a units annotation is worth the most: the physics of each declared
role is visible at a glance, and a dimensionally inconsistent equation, which
solves without complaint since the numbers are just numbers, is surfaced at
declaration time in the view the user already checks. The `(inc)` marker found
a real bug in the first IDAES example built on drto: a stage cost adding
joules-squared to watts-squared behind bare float scale factors.

## Acceptance criteria

- A model declared without units renders byte-identically to today, in both
  the text and HTML views.
- On a unit-carrying model, every declaration kind renders its units: states,
  controls, disturbances, dynamics, costs, initial conditions, terminal
  constraint, and the steady-state target pairs.
- A constraint body with inconsistent units renders `(inc)` and nothing else
  changes: no warning, no exception, declaration and transforms unaffected.
- Compact rendering: an energy holdup shows `J`, a duty shows `W`, a material
  balance shows `mol/s`.
- The annotation never raises: any failure to determine units renders as
  dimensionless does, silently.
