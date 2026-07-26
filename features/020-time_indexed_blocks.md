# Time-indexed Blocks in the terminal segment

**Status:** ![draft](https://img.shields.io/badge/draft-lightgrey)

## Description

As a user of DRTO with an IDAES flowsheet (or any model that keeps per-time
structure in `Block(time)` members rather than time-indexed Vars), I want
`drto.infinite_horizon` to replicate that structure onto the terminal segment,
so that the transform produces the same square, consistent tail it produces
for a flat model instead of an overdetermined one.

Today's discovery classifies a referenced variable by its own component's
index. A variable inside `properties_out[t]` has component index
`('H2O', ...)`, not time; its time lives on the parent Block one level up, so
the walk reads it as time-invariant and shares the single `t=0` member with
every segment collocation point. On the dynamic IDAES CSTR that wires ~330
segment constraint references into `properties_out[0.0]`, takes the model
from dof 21 to dof -48, and ipopt exits `TOO_FEW_DOF` (gh #12).

The feature, in the transform's own terms:

- **Discovery climbs the parent chain.** A referenced component whose parent
  chain crosses a `BlockData` indexed by the declared time set is
  block-borne and time-varying. The existing at-own-time-point validation
  extends to the block's index: a replicated equation may only reference the
  member at the constraint's own time.
- **The horizon-end member is the template.** For each referenced
  time-indexed Block `B`, the segment gains `b.<local_name>`, a `Block(tau)`
  whose every member is a structural copy of `B[t_end]`: its Vars (domains,
  bounds, units, initial values from the `t_end` solution), its Constraints
  and Expressions as written. The tail continues from the horizon end, so
  the end member is the right snapshot.
- **Fixed stays fixed, at the end values.** A Var fixed in `B[t_end]` (feed
  conditions, a fixed volume) is fixed at that value in every tau copy: the
  tail holds the last specified inputs, the same way the tail holds the
  parameterized control's last value.
- **Params are shared, not copied.** A Param referenced from inside a member
  resolves to the original; parameter and property-parameter blocks are
  time-invariant and stay shared as-is.
- **References in replicated equations rewire.** A variable reached through
  `B[t]` in a control-volume equation maps to `b.<local_name>[tau].<path>`,
  through the same substitution map the flat components use. Constraints
  inside a copied member reference their own member's variables and copy
  self-contained.

Out of scope, rejected with a clear error rather than silently mishandled: a
time-indexed Block nested inside another time-indexed Block, and a referenced
member at a time other than the constraint's own point.

## Benefit hypothesis

This is the one structural gap between drto and the IDAES model library: the
declarations attach cleanly to a dynamic flowsheet and
`drto.dynamic_optimization` already solves one to optimality, but the
infinite-horizon tail, drto's distinguishing capability, silently builds a
broken model on exactly the models the framework exists to serve. Fixing
discovery at the parent-chain level fixes it for every package that uses the
`Block(time)` idiom, not IDAES specifically, since the transform never needs
to know whose blocks they are. Building the copies from the template member
carries units by construction (feature 019 displays them, gh #10 fixed the
flat components), so the segment stays dimensionally consistent from day one.

## Acceptance criteria

- The dynamic IDAES CSTR (saponification packages, feature-002 declarations)
  goes through `drto.infinite_horizon` and comes out square: degrees of
  freedom are preserved across the transform (up to the documented pin
  slacks), and ipopt solves the transformed model to optimality.
- No segment constraint references a main-model Block member: the
  `properties_out[0.0]` reference count on the segment is zero.
- The copied members carry the template's units, and the replicated
  equations are dimensionally consistent on the segment (the registry's
  units column shows them; no `(inc)` that the declared model did not
  already have).
- A model with no time-indexed Blocks builds a byte-identical segment to
  today: hicks and the existing suite are unaffected.
- Fixed Vars inside copied members are fixed at the `t_end` values in every
  tau copy; free Vars initialize from `t_end`.
- A nested time-indexed Block, and a reference to a member away from the
  constraint's own time point, raise the transform's usual descriptive
  ValueError naming the offending component.
- The registry's transformation log reports how many Blocks were replicated,
  alongside the existing algebraic-components count.
