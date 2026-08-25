# drto.info

**Status:** ![shipped](https://img.shields.io/badge/shipped-brightgreen)

## Description

As a user of DRTO, I want DRTO to keep a single record on my model of what I
have declared and which transformations have been applied, so that the
transformations can find my declared components, guard themselves against
invalid or repeated application, and compose with each other correctly,
and so that I can read back a clear, DRTO-aware view of what DRTO has done to
the model.

```python
import pyomo.environ as pyo
import drto

# ... declared model m (feature 002) ...

drto.info(m)   # the registry: declarations by kind, transformations applied
# displaying it renders the DRTO-aware view: components grouped by role,
# compact symbolic equations, and each applied transformation's outcome

pyo.TransformationFactory("drto.dynamic_to_steady_state").apply_to(m)
drto.info(m)   # now also shows the reduction and what it did
```

On the hicks dynamic optimization of the examples, the notebook
rendering reads:

<div class="drto-registry"><b>drto registry</b><table><tbody><tr><td colspan="2">states: 2, controls: 2</td></tr><tr><td>horizon</td><td><code>t (ContinuousSet, 16 points)</code></td></tr><tr><td>states</td><td><code>zc (free), zt (free)</code></td></tr><tr><td>dynamics</td><td><code>dzc[t]  ==  (1 - zc[t])/(u2sf*v2[t]) - k0*zc[t]*exp(- ea/zt[t])  for t in t</code></td></tr><tr><td>dynamics</td><td><code>dzt[t]  ==  (ztf - zt[t])/(u2sf*v2[t]) + k0*zc[t]*exp(- ea/zt[t]) - a0*u1sf*v1[t]*(zt[t] - ztcw)  for t in t</code></td></tr><tr><td>controls</td><td><code>v1 (piecewise_constant, free), v2 (piecewise_constant, free)</code></td></tr><tr><td>tracking stage cost</td><td><code>cost[t]  ==  10*(zc[t] - zc_ss)**2 + 2*(zt[t] - zt_ss)**2 + (v1[t] - v1_ss)**2 + 0.5*(v2[t] - v2_ss)**2  for t in sorted(t)[:-1]</code></td></tr><tr><td>terminal cost</td><td><code>term  ==  10*(zc[5] - zc_ss)**2 + 2*(zt[5] - zt_ss)**2</code></td></tr><tr><td>initial conditions</td><td><code>zc[0]  ==  zc_hat</code></td></tr><tr><td>initial conditions</td><td><code>zt[0]  ==  zt_hat</code></td></tr><tr><td>steady-state targets</td><td><code>zc_ss (of zc), zt_ss (of zt)</code></td></tr><tr><td>steady-state control targets</td><td><code>v1_ss (of v1), v2_ss (of v2)</code></td></tr><tr><td>cost_group</td><td><code>drto_ih, drto_ih</code></td></tr></tbody></table><b>transformations</b><ol><li><code>drto.infinite_horizon: segment=3 elements x 5 Legendre points, beta=1.2, gamma=0.01563827, profile=collocation, horizon=kept, infinite tail appended, terminal_cost=terminal deactivated (the tail owns it), terminal=soft pin z(tau=1)=z_s on 2 states, mu=1000.0</code></li><li><code>drto.parameterize: profiles=v1 (piecewise_constant), v2 (piecewise_constant)</code></li><li><code>drto.build_objective: objective=sum of 28 weighted cost terms</code></li><li><code>drto.dynamic_optimization: horizon=kept, tracking_weight=(one stage cost declared), sensitivity=2 initial-condition Params declared</code></li></ol></div>

## Benefit hypothesis

As DRTO transforms a model, it frees and fixes variables, drops cost
terms, installs objectives, and appends segments, and none of that is
visible afterward except by inspecting components one by one. The
registry view shows the declarations by role and each applied
transformation with its outcome, so the user can read what DRTO has done
to the model and check that the problem about to be solved is the one
they meant.

DRTO already has to store the declarations somewhere for the
transformations to consume, so the record is nearly free, and using it
as the one source for re-application guards and composition makes those
checks and their error messages consistent across every transformation.
Backing it with Pyomo's namespaced private data keeps it isolated from
the user's component namespace and correct under model cloning, which
the `create_using` form of every transformation depends on.

## Acceptance criteria

- `drto.info(m)` returns the model's DRTO registry, creating it on first
  access. It is backed by `m.private_data('drto')`, so only DRTO's own code can
  write the `drto` scope and it never appears in the model's component tree.
- The registry records declarations, keyed by kind, and an ordered list of the
  transformations that have been applied to the model.
- Both renderings open with the problem's size, before the
  per-declaration lines (gh #63). The size is the number of declared
  states and controls, counted as members at one time point, so the
  count is the model's dimension and not its grid's. A model with
  neither declared shows no size line.
- A declaration records its target component in the registry. The
  transformations read the registry to find declared components rather
  than re-scanning the model.
- A transformation records itself in the registry when it is applied, and can
  query the registry for whether a given transformation has already run.
- `clone()` and `create_using` preserve the registry. A cloned model
  has its own independent registry, and every component reference stored
  in it is remapped to the clone's components, not the source model's.
- The applied declarations and the ordered list of applied
  transformations can be read back.
- A transformation deciding whether it may run reads the registry as the
  record of what has been applied, rather than inspecting the model, on
  the assumption that only DRTO transformations change the model's form.
  Where it is cheap, a transformation additionally cross-checks the model
  itself, for example that the objective it would build is not already
  present. If the user mutates the model outside DRTO between
  transformations, the outcome is not guaranteed.
- Displaying `drto.info(m)` renders a readable, DRTO-aware view of the model: a
  `__repr__` for the console and a `_repr_html_` for a Jupyter notebook panel,
  while its attributes stay queryable.
- The view groups components by role (the horizon or single point, states,
  controls marked free or fixed, dynamics, stage and terminal costs,
  initial conditions, steady-state targets), one labeled line each.
- Indexed constraints and the objective render in compact symbolic form: one
  equation per constraint family with a free index over its set, for example
  `dz[k]/dt == f(z[k], u[k])` for `k` in the time set, not the per-index
  expansion `pprint` produces.
- It annotates each applied transformation's outcome (what it freed or fixed,
  the terms it dropped, the objective it assembled, whether it kept or collapsed
  the horizon), read from the transformation log.
