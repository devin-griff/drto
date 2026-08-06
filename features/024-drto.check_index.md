# drto.check_index

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO, I want a function that checks whether my declared
model is an index-one DAE and names the offending variables and
constraints when it is not, so that a higher-index formulation is
caught as one error message before it surfaces as solver failures.

```python
import drto

# ... declared, discretized model m ...

report = drto.check_index(m)
report = drto.check_index(m, condition_limit=1e10)
```

The declarations define a semi-explicit DAE: the declared states and
their dynamics are the differential part, every other equality is the
algebra, and every variable that is not a state, a state's derivative,
or fixed data (controls, disturbances, parameters) is algebraic. The
model is index one exactly when the algebra's Jacobian with respect to
the algebraic variables is nonsingular. The check tests that condition
in two layers, both deterministic:

- **Structural**: a maximum matching of the algebraic constraints to
  the algebraic variables, on the incidence graph
  (``pyomo.contrib.incidence_analysis``). An unmatched variable means
  no algebraic constraint determines it — the system is structurally
  higher-index — and the Dulmage-Mendelsohn decomposition names the
  unmatched variables and constraints. This layer needs no values.
- **Numerical**: structure can pass while the values fail, so the
  Jacobian is evaluated at the model's current point and its condition
  is estimated. A condition estimate above ``condition_limit`` (default
  1e10) fails the check, and the decomposition's smallest blocks
  containing the near-singularity are named. This layer reads the
  model's values, so it runs after an initializer has filled the model;
  a model without values gets the structural verdict alone, and the
  report says the numerical layer was skipped.

The check assembles the algebra at one time point — the constraints
that apply pointwise, excluding the declared dynamics, the
discretization equations, and the initial conditions — the same
per-point system the cold start solves (feature 011). Unequal counts of
algebraic constraints and variables at that point are a descriptive
error naming the surplus side. When the structural layer fails, the
structural index is computed by Pantelides' algorithm and reported,
with the caveat stated in the report that the structural index can
disagree with the differentiation index when cancellation hides a
dependency.

The check writes nothing: no components added or removed, no values
changed. It returns a readable report in the feature 010 shape with the
verdict, the counts, and, on failure, the named members.

## Benefit hypothesis

A higher-index formulation does not announce itself: it surfaces as
undetermined variables at the first time point, near-dependent
constraint rows, stalled dual iterations, and solver failure labels
that point everywhere except the cause. The mixer-settler example spent
all of that before the transfer extents were identified by hand as
algebraic variables no algebraic constraint determines. This check
turns the same identification into one function call at declaration
time: the extent formulation fails structurally with the extents named,
and the reformulation on the reaction invariants passes.

## Acceptance criteria

- `drto.check_index(m)` on a declared, discretized index-one model
  returns a report with the index-one verdict, from a structurally full
  matching and a condition estimate below `condition_limit` at the
  current point.
- The mixer-settler stage posed on the reaction invariants
  (`examples/models/prommis_sx2.py`) passes. The MSContactor form
  (`examples/models/prommis_sx.py`) fails structurally, with the
  transfer extents and the dissociation extents among the named
  unmatched variables.
- On structural failure the report carries the structural index from
  Pantelides' algorithm and names the unmatched variables and
  constraints; on numerical failure it names the members of the
  near-singular blocks and the condition estimate.
- Unequal counts of algebraic constraints and algebraic variables raise
  a descriptive error naming the surplus side and its members.
- A model without values gets the structural verdict, and the report
  records that the numerical layer was skipped; filled values enable
  it. `condition_limit` is honored, and a non-positive value is a
  descriptive error.
- The check adds and removes nothing and changes no values, on the
  declared discretized model and after `drto.infinite_horizon`.
- A model with `Block(time)` structure checks the same way.
- Returns a readable report in the feature 010 shape.
