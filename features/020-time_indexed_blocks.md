# Time-indexed Blocks in the terminal segment

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

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
  replicated equation determines is the existing descriptive error, which is
  how a fixed feed surfaces and gets declared (disturbance or control), the
  same as any flat input.
- Time-invariant components, including Params and parameter blocks, are
  shared as-is, which is current behavior.

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
  inputs declared per the existing errors) keeps its degrees of
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
