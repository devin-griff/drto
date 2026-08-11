# Explicit NMPC

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want to sample my assembled optimization into a
labeled dataset, fit a neural network to the control policy, and evaluate
the fitted controller, so that the online solve can be replaced by a
function evaluation.

```python
import drto

# ... declared model m, transforms applied (the assembled optimization) ...

data = drto.explicit_nmpc_data(
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

policy = drto.explicit_nmpc_train(
    data,               # dataset, or a path
    validation=0.2,     # dataset, path, or a fraction split off `data`
    sobolev=True,       # gradient term in the loss
    gamma=1.0,
    hidden=(100, 100, 100),
    activation="tanh",
    lr=1e-3,
    schedule="cosine",  # "cosine" decays to lr_min; "flat" holds lr
    lr_min=1e-5,
    clip=1.0,           # gradient-norm cap; None disables
    epochs=50000,
    fine_tune=0.2,      # final fraction trained on the value error alone
    seeds=1,            # networks trained; the best by validation is kept
    device="auto",
)

u0 = policy({"ca_hat": 0.8, "cb_hat": 0.5, "tr_hat": 134.1, "tk_hat": 130.0})
policy.save("policy.pt")
policy = drto.ExplicitNMPC.load("policy.pt")

report = drto.explicit_nmpc_closed_loop(
    policy, m,
    samples=50,         # closed-loop steps
    x0=None,            # initial state; default the initial-condition Params' values
    disturbances=None,  # realization per step, as the simulations take it
    compare=True,       # also solve at each visited state; record the solver's control
)
```

`explicit_nmpc_data` draws the sampled Params from their box and labels
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
before each solve and left at its last solution.

`explicit_nmpc_train` fits the network on min-max-scaled inputs and
outputs, the scaling taken from the sampled boxes and the control
bounds, with the gradient labels scaled by the ranges. With `sobolev`
the loss adds `gamma` times the squared error of the network's Jacobian
against the stored derivatives; `fine_tune` trains the final fraction
of the budget on the value error alone. Every run trains the full
budget and keeps the weights with the best validation value error;
`seeds` trains that many networks and keeps the best by the same
metric. Training requires torch, an optional dependency, and a missing
install is a descriptive error.

The trained policy is callable with named input values in the model's own
units and returns the control values in theirs.
`explicit_nmpc_closed_loop` runs it closed loop against the declared model,
one `dynamic_simulation` step per sample. With `compare` the horizon
problem is also solved at each state the policy visits, so the report
carries one state trajectory and, per sample, the policy's applied
control beside the control the solver takes at that same state: the two
sequences are comparable point by point, and `drto.plot_controls` draws
them on one axes. The solver-driven closed loop is `drto.ideal_nmpc`,
not repeated here.

## Benefit hypothesis

An explicit controller replaces each online solve with a function
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

- `explicit_nmpc_data` draws by all three methods, seeded and
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
  non-optimal solve is recorded as a failure and excluded. `path`
  writes JSON that round-trips through the loader.
- `explicit_nmpc_train` honors every listed option; `sobolev=True` on a
  dataset without gradients is a descriptive error, and torch missing
  is a descriptive error naming the install, for training and for
  `ExplicitNMPC.load` alike. A `validation` fraction splits off the
  dataset reproducibly under the seed; a dataset or path supplied is
  used as given. Runs are reproducible under a seed, and with
  `seeds > 1` the network kept is the best by validation value error.
- The trained policy is callable with named inputs in model units and
  returns controls in model units; `save`/`load` round-trips it.
- `explicit_nmpc_closed_loop` steps the declared model under the policy and
  returns a readable report in the feature 010 shape holding the state
  and applied-control trajectory per sample, its summary stating the
  policy's closed-loop cost computed from that trajectory and stored
  nowhere else. `x0` and a supplied disturbance realization are
  honored, and the loop is deterministic without one. With `compare`
  the horizon problem is solved at each visited state and the report
  also carries the solver's control there: one state trajectory, two
  control sequences.
- `drto.plot_controls` on the closed-loop report draws the policy's
  applied control per sample and, when the report carries a `compare`
  run, the solver's control at the same states on the same axes, the
  two distinguished.
- The Klatt-Engell example's data generation, training, and figure are
  reproduced through these functions.
