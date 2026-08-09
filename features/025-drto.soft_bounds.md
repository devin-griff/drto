# drto.soft_bounds

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want a function that replaces variable bounds and
inequality constraints with the same relations carrying a nonnegative
slack, and penalizes those slacks in the objective, so that a problem
that stops at a point of local infeasibility returns a solution and the
nonzero slacks name the bounds it could not hold.

```python
import drto

# ... declared, discretized, initialized model m ...

report = drto.soft_bounds(m)                     # every bound and inequality
report = drto.soft_bounds(m, m.th_aq, m.c_aq)    # the named components only
report = drto.soft_bounds(m, weight=1e4)
```

For a variable bounded `lo <= x <= up`, the bound is removed from the
Var and written as two constraints, `x + s_lo >= lo` and
`x - s_up <= up`, with `s_lo` and `s_up` in `NonNegativeReals`. Both
ways of stating a bound are handled: the explicit `bounds` pair, and a
bound that comes from the domain, which is how the solvent extraction
stage states nonnegativity. Removing a domain's bound widens the domain
to `Reals`. A variable bounded on one side gets one slack; a fixed
variable gets none.

For an inequality constraint `g(x) <= b`, the constraint is rewritten in
place as `g(x) - s <= b`, and a two-sided inequality gets a slack per
side. Rewriting in place keeps the constraint's identity, so a
declaration that names it stays valid. Equality constraints are left
alone.

The slacks are added on the parent block of the variable or constraint
they relax, named `<name>_soft_up` and `<name>_soft_lo` and indexed as
that component is, so a model whose time structure is `Block(time)` is
handled the same as one whose components are indexed by time.

The penalty is `weight` times the sum of every slack, registered on the
registry as a cost group, which is the same path the infinite horizon's
endpoint pin uses. `soft_bounds` then calls `drto.build_objective`, so a
model whose objective was already installed carries the penalty
afterward. `weight` defaults to 1000 and is stored as a mutable Param on
the model, so a different penalty is one `set_value` away with no second
call. A model whose objective was the constant zero, which is what the
simulation transforms install, has the penalty alone as its objective:
that model minimizes its total bound violation.

Components named in the call restrict what is softened to those
components; with none named, every bounded variable and every inequality
on the model is softened. Either way two things are skipped: the
infinite horizon endpoint pin slacks and pin constraints, whose
nonnegativity is what makes that pin's split penalty an L1 norm, and the
slacks a previous `soft_bounds` call wrote, so a second call softens
only what the first did not.

The function returns a readable report in the feature 010 shape with the
counts of bounds and constraints softened, and records the same counts
as a transformation on the registry, which `drto.info(m)` renders.

## Benefit hypothesis

A solver that stops at a point of local infeasibility reports the
constraint violations at the point it stopped, which is a list of
whatever was worst when it gave up, not a statement of what the problem
cannot satisfy. Softening the bounds turns that into an answer: the
problem always has a solution, and the slacks that come back nonzero are
exactly the bounds that could not hold, with the amount by which each
one failed.

Many bounds in a flowsheet are stated for the solver's benefit rather
than by the physics. A concentration is declared nonnegative because a
negative one is meaningless, not because the model would be wrong at
minus 1e-12; a holdup fraction is bounded away from zero and one to keep
an interior-point method off the singularity. A bound of that kind
failing an entire solve is a poor trade, and a penalized slack makes the
trade explicitly: the solve continues, and the penalty says how much the
bound was worth.

The infinite horizon problem on the solvent extraction stage is where
this is measured. It carries bounds on every concentration, flow, and
holdup fraction, and its dynamic optimization has stopped at a point of
local infeasibility from initial conditions far from the setpoint.

## Acceptance criteria

- After `drto.soft_bounds(m)`, no unfixed variable other than the slacks
  carries a finite bound, and every bound that was removed is held by a
  constraint carrying a nonnegative slack. A solution whose slacks are
  all zero satisfies every original bound.
- A bound stated through the domain is softened like an explicit
  `bounds` pair: `NonNegativeReals` becomes `Reals`. A one-sided bound
  gets one slack. A fixed variable gets none.
- An inequality constraint is rewritten in place with a slack and keeps
  its identity, so a declaration naming it stays valid; a two-sided
  inequality gets a slack per side; an equality constraint is unchanged.
- The objective after the call is the objective before it plus `weight`
  times the sum of every slack. `weight` defaults to 1000, and setting
  the Param's value changes the penalty without a second call.
- Named components restrict the action to those components, and their
  bounds alone are softened.
- On a model carrying a terminal segment, the endpoint pin slacks and
  pin constraints are unchanged, and a second `soft_bounds` call leaves
  the slacks the first one wrote unchanged.
- A model whose objective was the constant zero has the penalty as its
  objective after the call, and solving it minimizes the total
  violation.
- A model carrying a bound that cannot hold solves to a nonzero slack on
  that bound and to the value the bound forbade; the same model with the
  bound satisfiable solves with every slack at zero and reaches the same
  solution as the model without the call.
- The call works on a model with `Block(time)` structure and on a model
  after `drto.infinite_horizon`.
- Returns a readable report in the feature 010 shape, and the counts are
  recorded as a transformation that `drto.info(m)` renders.
