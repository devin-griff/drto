# drto.info

**Status:** ![ready](https://img.shields.io/badge/ready-blue)

## Description

As a user of DRTO, I want DRTO to keep a single record on my model of what I
have declared and which transformations have been applied, so that the
transformations can find my declared components, guard themselves against
invalid or repeated application, and compose without stepping on each other,
and so that I can read back a clear, DRTO-aware view of what DRTO has done to
the model.

## Benefit hypothesis

DRTO already has to store the declarations somewhere for the transformations to
consume, so folding an ordered log of applied transformations into that same
registry is nearly free, and it turns the ad-hoc, per-transformation markers
into one uniform mechanism. A single source of truth makes re-application
guards and composition consistent across every transformation, and gives clear,
model-aware error messages. Backing it with Pyomo's namespaced private data
keeps it isolated from the user's component namespace and correct under model
cloning, which the `create_using` form of every transformation depends on.

## Acceptance criteria

- `drto.info(m)` returns the model's DRTO registry, creating it on first
  access. It is backed by `m.private_data('drto')`, so only DRTO's own code can
  write the `drto` scope and it never appears in the model's component tree.
- The registry records declarations, keyed by kind, and an ordered list of the
  transformations that have been applied to the model.
- A declaration records its target component in the registry; the
  transformations read the registry to find declared components rather than
  re-scanning the model.
- A transformation records itself in the registry when it is applied, and can
  query the registry for whether a given transformation has already run.
- The registry survives `clone()` and `create_using`: a cloned model has its
  own independent registry, and every component reference stored in it is
  remapped to the clone's components, not the source model's.
- The registry is inspectable: the applied declarations and the ordered list of
  applied transformations can be read back.
- Guarding uses the registry as the record of what DRTO has applied, on the
  assumption that DRTO drives a model of fixed form. Where it is cheap, a
  transformation additionally cross-checks the model itself, for example that
  the objective it would build is not already present. If the user mutates the
  model outside DRTO mid-pipeline, the outcome is not guaranteed.
- Displaying `drto.info(m)` renders a readable, DRTO-aware view of the model: a
  `__repr__` for the console and a `_repr_html_` for a Jupyter notebook panel,
  while its attributes stay queryable.
- The view groups components by role (the horizon or single point, states,
  controls marked free or fixed, dynamics, stage and terminal costs, boundary
  conditions, steady-state targets), one labeled line each.
- Indexed constraints and the objective render in compact symbolic form: one
  equation per constraint family with a free index over its set, for example
  `dz[k]/dt == f(z[k], u[k])` for `k` in the time set, not the per-index
  expansion `pprint` produces.
- It annotates each applied transformation's outcome (what it freed or fixed,
  the terms it dropped, the objective it assembled, whether it kept or collapsed
  the horizon), read from the transformation log.
