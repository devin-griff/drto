# Updating the MHE arrival cost

**Status: research note for the estimation modes, which are unspecced.** No
code depends on this yet. It records how the arrival cost must be updated, the
shortcut that looks right and is not, and the questions the spec has to settle.

## What the arrival cost is for

Moving horizon estimation truncates full-information estimation to a window of
`N`. The arrival cost carries everything before the window, so the truncation
is not a loss of the past. Exactly,

    Gamma_k(z) = the optimal cost of the full-information problem over [0, k],
                 subject to x_k = z.

With `Gamma_k` exact, MHE returns the full-information estimate. Every
practical scheme approximates it, and the approximation is where the design
decisions are.

## The update folds in exactly one stage

When the window advances by one, one stage leaves it. The recursion is

    Gamma_{k+1}(z) = min over x_k of [ Gamma_k(x_k) + L_k(x_k, w_k) ]
                     subject to  z = f(x_k, w_k)

where `L_k` is the stage cost of the interval leaving the window: the process
noise penalty on `w_k` and the measurement residual at `t_k`. The stages
`L_{k+1} ... L_{k+N-1}` stay inside the next window's objective. They are not
folded in.

## The shortcut that double counts

The window NLP has just been solved and its KKT factor is available, so it is
tempting to reduce that factor onto the new arrival state and call the result
`Pi^{-1}`. That reduces over every variable except the arrival state, which
includes the stages still live in the next window.

Scalar linear-Gaussian window `{0, 1, 2}` with `x_{j+1} = a x_j + w_j`,
`a = 0.9`, process information `Q^-1 = 1`, measurement information `R^-1 = 4`,
prior information `2`, reducing onto `x1`:

| | information on `x1` |
|---|---|
| one-step recursion | 0.8811 |
| full-window reduction | 5.5291 |

The larger number decomposes exactly:

    5.5291  =  0.8811  +  4.0000  +  0.6480
               arrival    measurement   the x2
               cost       at x1         stage

and the second and third terms are precisely what remains live in window
`k+1`'s objective. They get counted twice, once as a weight inside the arrival
cost and once as residuals. Here that is 6.28x over-confident, and the factor
compounds with the horizon, because every additional overlapping stage is
another term counted twice.

The estimator therefore builds the one-step subproblem (the previous arrival
cost plus the one departing stage, with the new arrival state as the free
endpoint) and reduces that. pounce supplies the reduction; choosing what to
reduce is drto's.

## Filtering versus smoothing

There is a real choice underneath the recursion, and the spec has to make it
deliberately.

In the recursion above, `x_k` sits at the value the window-`k` solve gave it,
and that solve saw measurements through `t_{k+N}`. So the information folded
forward is smoothed, conditioned on data later than `t_k`. Building
`Gamma_{k+1}` instead from data through `t_{k+1}` only gives the filtering
update, which is what a Kalman or extended-Kalman covariance recursion
produces.

Both appear in the MHE literature and they are not interchangeable: the
smoothing update is tighter, the filtering update is the one whose stability
arguments are standard. Rao, Rawlings and Mayne is the usual reference for
constrained MHE and its arrival-cost approximations. I have not re-read it in
detail, so the spec should check which update its stability argument assumes
before committing.

## A bound-active arrival state

If the arrival state sits at one of its bounds in the window just solved, three
candidate weights exist for that direction: zero, the barrier's `z^2 / mu`, and
the retained row of the reduced Hessian with the barrier diagonal removed. Only
the third is finite, and it is the one that answers the question the arrival
cost asks, how the past cost curves as that state moves off the bound into the
interior.

Two traps here.

**A bound is one-sided; a missing information row is two-sided.** The
reasoning that the next window re-imposes the bound anyway, so the arrival cost
need not carry that direction, does not hold. The bound blocks the infeasible
side. On the feasible side an absent row leaves no penalty at all, so the next
window can revise the arrival state far into the interior on weak data, despite
the previous window having had information about it.

**The quadratic form cannot carry the slope.** `Gamma(x0) = 0.5 (x0 - xhat)^T
Pi^{-1} (x0 - xhat)` has zero gradient at `xhat` by construction. At a bound
with a nonzero multiplier the true cost-to-arrive has a nonzero slope there, so
the form cannot represent it. Either the arrival cost gains a linear term, or
`xhat` is the unconstrained minimizer of the past problem (which lies outside
the bound) rather than the truncated solution, and the next window's own bound
does the truncating. The second is the statistically honest one: carry the
untruncated posterior forward and let the next window truncate it.

## Questions the spec has to settle

- Filtering or smoothing update, and which the stability argument assumes.
- What `xhat` is, and where the loop gets it. The truncated solution and the
  untruncated minimizer differ exactly when a bound is active, which is the
  case that matters most.
- What to do when the reduced Hessian is indefinite. The Lagrangian form can
  be, and an indefinite `Pi^{-1}` makes the next window's NLP unbounded below
  along the negative-curvature direction. Gauss-Newton stays PSD and is the
  obvious fallback, but that is a choice, not a default.
- Active-set churn between windows. `Pi^{-1}` changes rank as bounds activate
  and release, and near a weakly active bound the classification itself is
  unstable.
- Whether an EKF-style covariance recursion is offered as a cheap alternative
  to reducing a subproblem each step.

## Relation to pounce

The reduction primitive is `information()` in pyomo-pounce, specified in that
repo's `dev-notes/covariance-information-roadmap.md`. The split is that pounce
reduces a held factor onto a block and reports the activity classification with
it; drto decides what to reduce, what the arrival cost weights, and what to do
when the active set moves. The one pounce-side requirement this note depends on
is that a pinned direction reports its finite retained-row value rather than a
zero, since that number cannot be recovered downstream.
