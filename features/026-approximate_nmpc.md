# Approximate NMPC

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want to sample my assembled optimization into a
labeled dataset, fit a neural network to the control policy, and evaluate
the fitted controller, so that the online solve can be replaced by a
function evaluation.

```python
import drto

# ... declared model m, transforms applied (the assembled optimization) ...

data = drto.approximate_nmpc_data(
    m,
    n=1000,             # draws
    method="sobol",     # "sobol" | "lhs" | "uniform"
    inputs=None,        # Params to sample; default the initial-condition Params
    ranges=None,        # {param: (lo, hi)}; default the paired state's bounds
    gradients=True,     # store du0/dinput, read from the solve's factorization
    solver="pounce",
    seed=0,
    path=None,          # JSON written when given
)

policy = drto.approximate_nmpc_train(
    data,               # dataset, or a path
    validation=0.2,     # dataset, path, or a fraction split off `data`
    validation_loss="sobolev",  # "sobolev" | "value": the checkpoint metric
    training_loss="sobolev",    # "sobolev" | "value": the loss minimized
    gamma=1.0,
    hidden=(100, 100, 100),
    activation="tanh",
    steady_state_enforced=True,  # the policy returns the recorded steady
                        # control at the recorded steady state exactly
    lr=1e-3,
    schedule="cosine",  # "cosine" decays to lr_min; "flat" holds lr
    lr_min=1e-5,
    clip=1.0,           # gradient-norm cap; None disables
    weight_decay=0.0,   # AdamW's decoupled L2 penalty on the weights; 0 disables
    epochs=50000,
    fine_tune=0.2,      # final fraction trained on the value error alone
    seeds=1,            # networks trained; the best by validation is kept
    device="auto",
)

u0 = policy({"ca_hat": 0.8, "cb_hat": 0.5, "tr_hat": 134.1, "tk_hat": 130.0})
policy.save("policy.pt")
policy = drto.ApproximateNMPC.load("policy.pt")

report = drto.approximate_nmpc_closed_loop(
    policy, m,
    samples=50,         # closed-loop steps
    x0=None,            # initial state; default the initial-condition Params' values
    disturbances=None,  # realization per step, as the simulations take it
    solver="pounce",
    compare=True,       # also solve at each visited state; record the solver's control
)

axes = drto.plot_history(policy)   # the run's two loss curves
axes = drto.plot_parity(policy, data)  # each control against the solver's label
```

`approximate_nmpc_data` draws the sampled Params from their box and labels
each draw with one solve: the first control action, the objective, and,
with `gradients`, the action's derivative with respect to every sampled
Param, read from the converged factorization the way
`drto.advanced_step_controller` reads it. The default box comes from the
bounds of the state each initial-condition Param pins; `ranges` narrows
or widens it per Param. `method` picks the design: `sobol` (the
default) stays evenly spread in any leading subset, so one pool serves
nested training-set sizes; `lhs` stratifies every coordinate exactly at
its own `n`; `uniform` is independent draws, the sampling of Lueken,
Brandner & Lucia (2023). Draws whose solve does not return optimal are
recorded as failures and carry no labels. The model is cold-started
before each solve and left at its last solution. When the model
declares `steady_state` and `steady_state_control`, the dataset
records the targets' values as `x_ss` and `u_ss`, which
`steady_state_enforced` training reads.

`approximate_nmpc_train` fits the network on min-max-scaled inputs and
outputs. The inputs scale by the sampled boxes. Each control scales
by the span its labels occupy over the training set, floored at one
thousandth of the control's bound range, so a control commanding a
sliver of its box still trains at full price. A supplied validation
set measures in the training set's units, the gradient labels scale
accordingly, and the policy records the scale. `training_loss`
picks the loss the run minimizes: `"sobolev"`, the default, adds
`gamma` times the squared error of the network's Jacobian against the
stored derivatives to the value error, and `"value"` fits the values
alone. `fine_tune` trains the final fraction
of the budget on the value error alone. `weight_decay` passes AdamW's
decoupled L2 penalty on the weights, applied in the Sobolev and
fine-tune phases alike. Every run trains the full
budget and keeps the weights with the best validation loss.
`validation_loss` picks the metric the checkpoints and the `seeds`
winner are judged by. `"sobolev"`, the default, evaluates the
training loss on the validation set (the value error plus `gamma`
times the gradient term, weighted alike), one definition across both
phases. `"value"` is the value error alone, the metric the paper
reports. The choice is independent of `training_loss`. `seeds` trains
that many networks and keeps the best by the same metric. The kept policy
carries its run's per-epoch training loss and validation loss as
`policy.history`, and, when the validation set was split off by
fraction, the points it was split to as `policy.meta`'s
`validation_index`, so a caller can tell the two sets apart without
reproducing the split.

With `steady_state_enforced`, the default, the policy is the network
plus a constant offset: u(x) = u_ss + N(x) - N(x_ss), in the scaled
units. The two network terms cancel at the recorded steady state, so
the policy returns the recorded steady control there exactly,
whatever the weights, and the closed loop has the declared
equilibrium as a fixed point. The Jacobian is the network's, so the
gradient labels and the Sobolev terms are untouched. A dataset
without the recorded `x_ss` and `u_ss` under this option is a
descriptive error naming regeneration. Training requires torch, an
optional dependency, and a missing install is a descriptive error.

The trained policy is callable with named input values in the model's own
units and returns the control values in theirs.
`approximate_nmpc_closed_loop` runs it closed loop against the declared model,
one `dynamic_simulation` step per sample. Each action is clamped to the
declared control's bounds before it is applied, the way an actuator
holds an out-of-range command at its limit, and the report's moves are
the applied values. The plant clone sheds the domain and bounds of
every variable the loop does not fix. A square simulation cannot
steer away from a bound, so the step converges wherever the dynamics
land and the report records an excursion outside the controller's
box. A plant step that still fails raises an error naming the visited
state and the applied action. With `compare` the horizon
problem is also solved at each state the policy visits, so the report
carries one state trajectory and, per sample, the policy's applied
control beside the control the solver takes at that same state: the two
sequences are comparable point by point, and `drto.plot_controls` draws
them on one axes. A visited state the horizon problem does not solve
at records no solver control for that sample, and the loop keeps
stepping the plant. The solver-driven closed loop is `drto.ideal_nmpc`,
not repeated here.

## Benefit hypothesis

A fitted controller replaces each online solve with a function
evaluation, so NMPC can run where a solver cannot: an embedded target,
a sampling rate the solve cannot meet, or a platform that carries no
solver. The offline cost is the labeled dataset, one solve per point,
and it grows with the model size and the input dimension. Both data
options lower that cost: each solve's factorization also yields the
derivative of the action with respect to every sampled input, so the
gradient labels multiply the information per solve at no additional
solves, and their value is largest exactly where solves are expensive
and the dataset is therefore sparse; the low-discrepancy designs
spend a fixed solve budget evenly over the operating region. The
training defaults are the configuration this package measured best,
and the options reproduce the protocol of Lueken, Brandner & Lucia
(2023) exactly when set to it.

## Acceptance criteria

- `approximate_nmpc_data` draws by all three methods, seeded and
  reproducible. The first `m` points of a Sobol pool of `n > m` are the
  same as a Sobol pool of `m`; an LHS of `n` has exactly one point in
  each of `n` equal bins per coordinate. Default ranges come from the
  pinned states' bounds; `ranges` overrides per Param; `inputs` extends
  the sampled set beyond the initial-condition Params.
- Each labeled point carries the sampled values, the first control
  action per declared control, the objective, and, with `gradients`,
  the derivative of each first action with respect to each sampled
  Param. The derivatives are read from the pounce factorization, so
  `gradients=True` with any other solver is a descriptive error. A
  model declaring `steady_state` and `steady_state_control` has the
  targets' values recorded in the dataset as `x_ss` and `u_ss`. A
  non-optimal solve is recorded as a failure and
  excluded. `path` writes JSON that round-trips through the loader.
- `approximate_nmpc_train` honors every listed option; the `"sobolev"`
  training or validation loss on a dataset without gradients is a
  descriptive error, and torch missing
  is a descriptive error naming the install, for training and for
  `ApproximateNMPC.load` alike. A `validation` fraction splits off the
  dataset reproducibly under the seed; a dataset or path supplied is
  used as given. The controls scale by their labels' span over the
  training set, recorded in the policy's metadata, with a
  near-constant control floored at one thousandth of its bound
  range. `weight_decay` applies AdamW's decoupled penalty in
  both phases; the default `0.0` reproduces the unpenalized run. Runs
  are reproducible under a seed, and with `seeds > 1` the network
  kept is the best by the validation loss. `validation_loss` honors
  both choices. Under `"sobolev"` the metric adds `gamma` times the
  gradient error, and a supplied validation dataset without gradient
  labels is a descriptive error. Under `"value"` such a dataset
  trains. With `steady_state_enforced` the policy returns the
  recorded steady control at the recorded steady state exactly, the
  equality survives `save`/`load`, and a dataset without the recorded
  targets is a descriptive error.
- The trained policy is callable with named inputs in model units and
  returns controls in model units; `save`/`load` round-trips it, the
  training history included.
- `approximate_nmpc_closed_loop` steps the declared model under the policy and
  returns a readable report in the feature 014 shape, the recorded
  bounds included, holding the state
  and applied-control trajectory per sample, its summary stating the
  policy's closed-loop cost computed from that trajectory and stored
  nowhere else. `x0` and a supplied disturbance realization are
  honored, and the loop is deterministic without one. An action outside
  a control's bounds is clamped to them before the plant step, and the
  recorded move is the applied value. The plant simulation carries no
  state bounds, and a policy that drives a state past the controller's
  bound simulates, the report recording the excursion. With `compare`
  the horizon problem is solved at each visited state and the report
  also carries the solver's control there: one state trajectory, two
  control sequences. A compare solve that does not return optimal
  records nan for that sample and the sample's time, and the run
  continues.
- `drto.plot_states` draws the closed-loop report's state trajectory
  directly, the way it draws the loop histories. `drto.plot_controls`
  on the report draws the policy's applied control per sample and, when
  the report carries a `compare` run, the solver's control at the same
  states on the same axes, the two distinguished. A sample the compare
  solve failed at is marked on the applied trajectory and named in the
  legend.
- `drto.plot_history` draws the kept run's training and validation
  losses on one log-scaled panel against the epoch each checkpoint was
  taken at.
- `drto.plot_parity` draws one panel per control, the label on the
  horizontal axis and the policy's action on the vertical, with the
  line where the two agree. Training and validation points draw in
  different markers and each series carries its coefficient of
  determination. The split comes from `validation_index`, or from a
  supplied validation dataset, and without either one series draws.
- The Klatt-Engell example's data generation, training, and figure are
  reproduced through these functions.
