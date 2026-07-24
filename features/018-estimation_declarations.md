# Estimation declarations

**Status:** ![implemented](https://img.shields.io/badge/implemented-yellowgreen)

## Description

As a user of DRTO, I want to declare the estimation pieces of my moving-horizon
estimation problem, the parameters I am estimating, the process noise, the
measurements, and the estimation costs, the same way I declare the control
pieces, so that DRTO can assemble the estimation objective without my
restructuring the model or writing a separate estimation model.

Moving horizon estimation is the dual of the control problem, so the surface is
the one from feature 002: tag a component you already built, or wrap a fresh one
as you build it, with the cost constraints doubling as decorators. The example
tags an MHE window over a first-order system with an unknown rate constant.

```python
import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
import drto

m = pyo.ConcreteModel()
N, h = 10, 1  # window samples and sampling time
m.t = ContinuousSet(initialize=pyo.RangeSet(0, N*h, h))  # the window grid
drto.horizon(m.t)

m.z = pyo.Var(m.t)
drto.state(m.z)
m.dzdt = DerivativeVar(m.z, wrt=m.t)

m.k = pyo.Var(initialize=1.0)   # unknown rate constant, constant over the window
drto.estimated_parameter(m.k)

m.w = pyo.Var(m.t)              # process noise in the dynamics
drto.disturbance(m.w)

m.y_meas = pyo.Param(m.t, mutable=True, initialize=0.0)  # measurements over the window
drto.measurement(m.y_meas)

@m.Constraint(m.t)
def ode(m, t):
    return m.dzdt[t] == -m.k*m.z[t] + m.w[t]
drto.dynamics(m.ode)

# inverse covariances: the cost weights
m.inv_r = pyo.Param(initialize=100, mutable=True)   # measurement
m.inv_q = pyo.Param(initialize=10, mutable=True)    # process noise
m.inv_p0 = pyo.Param(initialize=1, mutable=True)    # arrival
m.z_prior = pyo.Param(initialize=0.4, mutable=True)  # arrival prior mean

m.est_stage = pyo.Var(m.t)

@m.Constraint(sorted(m.t)[:-1])  # the terminal term owns the final time
def est_stage_con(m, t):
    return m.est_stage[t] == m.inv_r*(m.y_meas[t] - m.z[t])**2 + m.inv_q*m.w[t]**2
drto.estimation_stage_cost(m.est_stage_con)

m.est_term = pyo.Var()

@m.Constraint()
def est_term_con(m):
    tN = m.t.last()
    return m.est_term == m.inv_r*(m.y_meas[tN] - m.z[tN])**2
drto.estimation_terminal_cost(m.est_term_con)

m.arrival = pyo.Var()

@m.Constraint()
def arrival_con(m):
    return m.arrival == m.inv_p0*(m.z[0] - m.z_prior)**2
drto.arrival_cost(m.arrival_con)
```

The wrap and decorator forms work exactly as in feature 002: each function
returns a fresh component for the `m.x = ...` assignment, and the constraint-role
declarations (`estimation_stage_cost`, `estimation_terminal_cost`,
`arrival_cost`) double as decorators taking the model plus what `@m.Constraint`
would take. The styles mix per component.

## Benefit hypothesis

MHE is the dual of the control problem: the same states and dynamics, a cost
over a window, solved as one NLP. Declaring its pieces through the same surface
as the control side lets one Pyomo model carry both the control and the
estimation problem, so a single declared model serves every mode. Recording the
estimation declarations in `drto.info` gives the estimation transforms one place
to find them, the way `build_objective` and the control transforms consume the
control-side declarations, and it lets `drto.dynamic_optimization` (feature 006)
recognize and set aside the estimation pieces when it assembles a control
problem from the same model.

## Acceptance criteria

- Each estimation declaration follows the feature-002 surface: it tags an
  existing Pyomo component or wraps a fresh one, validates the target's type and
  convention with a clear error, and records it in `drto.info(m)` (feature 001)
  under a kind equal to the function name. The styles mix per component, and the
  ordering rule is the same, a declaration's prerequisites must be declared by
  the time it registers.
- The constraint-role declarations (`estimation_stage_cost`,
  `estimation_terminal_cost`, `arrival_cost`) double as decorators taking the
  model plus whatever `@m.Constraint` would take, building, attaching, and
  declaring the constraint in one step.
- Arity: `estimated_parameter`, `disturbance`, and `measurement` accept varargs
  or an indexed container (one declaration per container), since they scale with
  the estimated quantities. `estimation_stage_cost`, `estimation_terminal_cost`,
  and `arrival_cost` each take exactly one object and error on more.
- Re-declaration: the varargs declarations accumulate across calls and reject
  declaring the same component twice as a duplicate. The single-object cost
  declarations error on a second call with a different object, since the model
  has one of each. Both checks run against the registry (feature 001).
- `estimated_parameter(m.theta, ...)` tags one or more Vars for unknown model
  parameters to estimate, constant over the window, so they are not indexed by
  the horizon time set. This declaration is shared with the steady-state
  data-reconciliation mode, so it does not require a declared horizon.
- `disturbance(m.w, ...)` tags one or more Vars used as process noise, the free
  variables the estimator adjusts to reconcile the model with the data,
  penalized by their inverse covariance in the estimation stage cost. How the
  noise enters the model equations is up to the user, not fixed by the
  declaration. Process noise is not a manipulated input: it takes no `profile`
  and is unrelated to `control` (USER DECISION 2026-07-14). With a horizon
  declared, each disturbance is indexed by the declared time set, checked at the
  declaration.
- `measurement(m.y_meas, ...)` tags one or more mutable Params holding the
  measured values over the window, which drto refreshes each step like the state
  feedback hook. They appear in the estimation cost residuals
  `||y_meas - h(z)||`; `h(z)` is written inline in the cost, so there is no
  output Var or defining constraint to tag (USER DECISION 2026-07-14). With a
  horizon declared, each measurement is indexed by the declared time set.
- `estimation_stage_cost(m.con)` tags a per-time-point equality Constraint whose
  left-hand side is the scalar running estimation-cost variable, the measurement
  residual plus the process-noise penalty, each weighted by an inverse
  covariance. Like `tracking_stage_cost`, it is indexed over the sample grid
  minus the final point, one member per sample: the process noise leads out of
  each point to the next and nothing leads out of the final point, so the final
  time belongs to the terminal term. A member at the final time or a missing
  sample is rejected, and the family may not be indexed by the time set itself.
- `estimation_terminal_cost(m.con)` tags an equality Constraint whose left-hand
  side is the scalar terminal estimation-cost variable, the current-state
  measurement residual at the final (present) time with no process noise. It is
  a distinct terminal term rather than part of the stage sum, a standard MHE
  term (USER correction 2026-07-14).
- `arrival_cost(m.con)` tags an equality Constraint whose left-hand side is the
  scalar arrival-cost variable, the soft prior on the window's initial state
  `||z(t0) - z_prior||` referencing declared states at the first time point. It
  is the dual of the control-side `initial_condition` but soft, a cost rather
  than a hard equality. Its weight is the arrival-cost inverse covariance, the
  piece the covariance propagation updates each step (the Gauss-Newton pounce#203
  machinery, a follow-on).
- The scalar-side conventions of the three cost constraints are read from the
  written equality's sides, either orientation, so `lhs == rhs` and `rhs == lhs`
  are equivalent; each must be written as an explicit equality.
- This feature is the declaration surface only. The estimation mode transforms
  that assemble the estimation objective from these declarations (moving horizon
  estimation, and steady-state data reconciliation) are a follow-on. The
  registered kinds are what `drto.dynamic_optimization` (feature 006) sets aside
  when it assembles a control problem from a model that also carries estimation
  declarations.
