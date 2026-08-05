# Time-indexed Blocks in the terminal segment

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO with an IDAES flowsheet (or any model that keeps per-time
structure in `Block(time)` members rather than time-indexed Vars), I want
`drto.infinite_horizon` to treat that structure the way it already treats
flat components, so the transform stays degree-of-freedom neutral on these
models the way it is on flat ones: the tail adds unknowns and equations in
equal number, and the model keeps exactly the freedom it had, its control
moves.

Discovery classifies a referenced variable by its own component's index. A
variable inside `properties_out[t]` is indexed by component, not by time; its
time lives on the parent Block, which the walk never looks at. So the walk
calls it time-invariant and shares the single `t=0` member with every segment
collocation point: ~330 segment references into `properties_out[0.0]` on the
dynamic IDAES CSTR, and a tail that destroys degrees of freedom instead of preserving them,
21 before the transform to -48 after (gh #12).

The feature is the discovery fix, with everything downstream unchanged:

- A referenced variable inside a `Block(time)` member is time-varying. It
  joins the same classification the flat scan applies: a declared state or
  control by identity, otherwise algebraic.
- Block-borne algebraic variables get segment copies exactly as flat ones
  do: new Vars over `tau`, units from the source (feature 019 / gh #10), no
  values copied from any time point.
- The equations inside the members are algebraic equations. Each family
  (same constraint name across the members) replicates onto the segment from
  a representative member, exactly as flat algebraic constraints do.
- The existing validations apply unchanged: a reference away from the
  constraint's own time point is rejected, and a copied variable that no
  replicated equation determines is the existing descriptive error.
- A fixed variable is a specification, not a decision: its segment copy is
  fixed at the horizon-end value, with no declaration involved. An IDAES
  feed, a fixed volume, or any given input carries onto the tail as itself.
  Declared disturbances remain zero-mean noise, which the optimization modes
  fix at zero on the horizon, so an absolute input must not be declared as
  one.
- Declared disturbances gain the tail handling the mode transforms already
  have, closing a hole this model exposes but any flat model with a declared
  disturbance hits today: the transform takes a `disturbances` option, and
  each declared disturbance's segment copy is fixed at the given constant,
  default zero. The tail continues under nominal disturbance unless told
  otherwise, mirroring `drto.dynamic_simulation`.
- Time-invariant components, including Params and parameter blocks, are
  shared as-is, which is current behavior.
- A Block family may carry index sets besides time (an IDAES stage element,
  a spatial node: `Block(t, s)`). Each non-time combination is its own
  family: its members get their own segment copies and its equations
  replicate per combination, the same treatment a partially declared
  container's members already get. The segment's copy names carry the
  combination.
- A derivative over a ContinuousSet other than the declared time set (a
  spatial axis) is ordinary algebra: its referenced members copy to the
  segment and its discretization equations, despite the pyomo.dae naming,
  replicate with them. Only the declared time set's discretization and
  continuity equations are the artifacts the segment rebuilds over tau.
- Two model components sharing a local name (the two settlers of a
  mixer-settler both carry `_flow_terms`) get distinct segment names, the
  sanitized full path breaking the tie.

Out of scope, rejected with a descriptive error: a time-indexed Block nested
inside another time-indexed Block.

## Benefit hypothesis

This is the one structural gap between drto and the IDAES model library: the
declarations attach cleanly to a dynamic flowsheet and
`drto.dynamic_optimization` already solves one to optimality, but the
infinite-horizon tail silently builds a broken model on exactly the models
the framework exists to serve. The fix is at discovery, so it applies to any
package using the `Block(time)` idiom, and everything after discovery reuses
machinery that is already tested.

## Acceptance criteria

- The dynamic IDAES CSTR (saponification packages, feature-002 declarations,
  no declarations beyond them) keeps its degrees of
  freedom across `drto.infinite_horizon`, up to the documented pin slacks, and ipopt solves the result.
- No segment constraint references a main-model Block member.
- Segment copies of block-borne variables carry their source units, and the
  replicated equations are dimensionally consistent on the segment.
- A model with no time-indexed Blocks builds a byte-identical segment to
  today: hicks and the existing suite are unaffected.
- A reference to a member away from the constraint's own time point, and a
  nested time-indexed Block, raise the transform's usual descriptive
  ValueError naming the offending component.
- The registry's transformation log counts the replicated Block families
  alongside the existing algebraic-components count.
- A flat model with a declared disturbance goes through the transform with
  the disturbance's segment copy fixed at the `disturbances` value (default
  zero), on a model with no Blocks at all: the hole is closed for the flat
  case too, and tested there.
- A `Block(t, s)` family replicates per non-time combination and solves
  identically to the flat twin of the same physics; a spatial
  discretization equation replicates as algebra with its derivative members
  copied; same-named components take distinct segment names; and the
  free-copy guard covers partially copied containers, so a member left
  without a replicated defining equation is the descriptive error, not a
  silent freedom. The declared PrOMMiS mixer-settler goes through the
  transform with its degrees of freedom accounted for.
