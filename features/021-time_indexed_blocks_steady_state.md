# Time-indexed Blocks in the steady-state reduction

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO with an IDAES flowsheet (or any model that keeps per-time
structure in `Block(time)` members rather than time-indexed Vars), I want
`drto.dynamic_to_steady_state` to collapse that structure the way it already
collapses flat components, so the declared dynamic model is the single source
of its own steady state: the setpoint comes from the reduction, not from a
second hand-built steady flowsheet kept in sync with the dynamic one.

The collapse walks time-indexed Vars. A variable inside `properties_out[t]`
is indexed by component, not by time; its time lives on the parent Block,
which the walk never looks at. So the member Blocks survive at every time
point: on the dynamic IDAES CSTR all 25 members of `properties_in`,
`properties_out`, and `reactions` remain, the collapsed balances reference
the `t=0` members, the other 24 stay behind as orphaned structure, and the
"steady" model comes out with 284 free variables against 95 active
constraints. pounce errors on it.

The feature is the collapse fix, the reduction treating a `Block(time)` as
the time-varying structure it is:

- A `Block(time)` family collapses to its single steady member: the `t=0`
  member stays, its variables and internal equations as written, and the
  other members leave the model with their contents. Nothing is rebuilt, so
  values, bounds, units, and fixed status carry through trivially.
- Fixed stays fixed: a fixed variable in the surviving member is a
  specification and remains fixed, so an IDAES feed holds at its value
  through the reduction with no declaration involved.
- A time-indexed Reference whose referents live in the members (an IDAES
  Port entry such as `inlet.flow_vol[t]`) collapses to a Reference onto the
  surviving member. It must not become a fresh independent Var: the Port
  keeps pointing at its referent.
- Time-invariant Blocks, including Params and parameter blocks, are shared
  as-is, which is current behavior.

Out of scope, rejected with a descriptive error: a time-indexed Block nested
inside another time-indexed Block, mirroring the terminal-segment rule
(feature 020).

## Benefit hypothesis

DESIGN.md's setpoint-consistency story says the steady target comes from the
same declared model the controller uses. On IDAES models that story is
broken today: the declarations attach cleanly and `drto.dynamic_optimization`
solves the flowsheet, but the reduction silently builds a broken model, so a
user hand-builds a `dynamic=False` twin flowsheet and keeps its feed and
configuration in sync by discipline. Feature 020 closed this gap for the
terminal segment; this closes it for the steady-state reduction, the last
transform whose discovery is blind to the `Block(time)` idiom. The fix is at
the collapse walk, so it applies to any package using the idiom, and
everything downstream reuses machinery that is already tested.

## Acceptance criteria

- The dynamic IDAES CSTR (saponification packages, feature-002 declarations,
  nothing beyond them) reduces to the steady system:
  `drto.steady_state_simulation` leaves it square, pounce solves it, and the
  solution matches the hand-built `FlowsheetBlock(dynamic=False)` model, the
  reactor at its feed-alone equilibrium.
- Exactly one member of each `Block(time)` family survives the reduction,
  and no active constraint or Reference reaches a removed member.
- A model with no time-indexed Blocks reduces to the same steady system as
  today: hicks and the existing suite are unaffected.
- A time-indexed Block nested inside another raises the transform's usual
  descriptive ValueError naming the offending component.
- The registry's transformation log counts the collapsed Block families
  alongside the existing Var and Constraint counts.
- The IDAES CSTR example notebook declares the flowsheet once and runs two
  solves total: the reduced model for the setpoint, then the
  infinite-horizon controller.
