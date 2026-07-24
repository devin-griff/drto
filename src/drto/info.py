# Copyright (c) 2026 Devin Griffith
# SPDX-License-Identifier: BSD-3-Clause
"""The drto registry: ``drto.info`` (feature 001).

One record per model of what has been declared and which transformations have
been applied. Declarations record their target components here and the
transformations read the registry to find them, rather than re-scanning the
model. The registry is backed by ``Block.private_data``, so it never appears
in the model's component tree and it survives ``clone()`` (and therefore
``create_using``) with every stored component reference remapped to the
clone's components.

Displaying the registry renders a drto-aware view of the model: components
grouped by role, indexed constraints in compact symbolic form (one equation
per constraint family with a free index over its set), and the ordered log of
applied transformations with their outcomes.
"""
import html
import inspect
import re

from pyomo.core.expr.template_expr import templatize_constraint

#: The ``Block.private_data`` scope drto stores its registry under.
_DRTO_SCOPE = "drto"

#: Display order and labels for the declaration kinds of the control-side
#: surface (feature 002) and the estimation-side surface (feature 018).
#: Unknown kinds render after these, labeled by their raw kind string, so the
#: registry does not need updating for new kinds.
_KIND_LABELS = (
    ("horizon", "horizon"),
    ("state", "states"),
    ("dynamics", "dynamics"),
    ("control", "controls"),
    ("tracking_stage_cost", "tracking stage cost"),
    ("economic_stage_cost", "economic stage cost"),
    ("tracking_terminal_cost", "terminal cost"),
    ("initial_condition", "initial conditions"),
    ("terminal_constraint", "terminal constraint"),
    ("steady_state", "steady-state targets"),
    ("steady_state_control", "steady-state control targets"),
    ("estimated_parameter", "estimated parameters"),
    ("disturbance", "disturbances"),
    ("measurement", "measurements"),
    ("estimation_stage_cost", "estimation stage cost"),
    ("estimation_terminal_cost", "estimation terminal cost"),
    ("arrival_cost", "arrival cost"),
)

#: Fallback free-index names, for rules whose own argument names are
#: unavailable.
_INDEX_NAMES = ("k", "j", "i", "l")

#: Free-index names for a rule's internal sums.
_SUM_INDEX_NAMES = ("i", "j", "l", "p", "q")


def _rule_index_names(con, count):
    """The rule function's own index argument names, in model order.

    Returns None when the rule is not a plain function of the model plus
    ``count`` indices (a lambda from data, a transformed family, an arity
    mismatch), and the fallback pool takes over.
    """
    rule = getattr(con, "rule", None)
    fcn = getattr(rule, "_fcn", rule)
    if not callable(fcn):
        return None
    try:
        params = list(inspect.signature(fcn).parameters)
    except (TypeError, ValueError):
        return None
    if len(params) != count + 1:
        return None
    return params[1:]


def _index_names(con, count):
    """Free-index names for a family: the rule's own, or the fallback pool."""
    return _rule_index_names(con, count) or [
        _INDEX_NAMES[n % len(_INDEX_NAMES)] for n in range(count)
    ]


def _name_sum_indices(s, outer_names):
    """Rename the template placeholders of a rule's internal sums.

    Templatization numbers every placeholder globally: the constraint's own
    indices first (renamed by the caller), then one per internal sum, left
    as ``_2``. Those get names from a pool that skips the names the
    constraint indices took.
    """
    pool = [x for x in _SUM_INDEX_NAMES if x not in outer_names] or list(
        _SUM_INDEX_NAMES
    )
    n_outer = len(outer_names)
    nums = sorted({int(x) for x in re.findall(r"(?<!\w)_(\d+)", s)}, reverse=True)
    for num in nums:
        name = pool[(num - n_outer - 1) % len(pool)]
        s = re.sub(rf"(?<!\w)_{num}(?!\d)", name, s)
    return s


def info(m):
    """Return ``m``'s drto registry, creating it on first access.

    Parameters
    ----------
    m : Block
        The model (or block) whose registry is requested.

    Returns
    -------
    Info
        The registry. Repeated calls return the same object, and a cloned
        model has its own independent registry with every stored component
        reference remapped to the clone.
    """
    store = m.private_data(_DRTO_SCOPE)
    reg = store.get("info")
    if reg is None:
        reg = store["info"] = Info()
    return reg


class Info:
    """The drto registry: declarations by kind plus a transformation log.

    Declarations are recorded by the declaration functions (feature 002)
    and read back by the transformations. Transformations record themselves,
    with an outcome annotation, in application order. Displaying the object
    (console ``repr`` or Jupyter) renders the drto-aware view of the model.
    """

    def __init__(self):
        self._declarations = {}
        self._transformations = []

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    def record_declaration(self, kind, component, **metadata):
        """Record a declaration of ``component`` under ``kind``.

        Parameters
        ----------
        kind : str
            The declaration kind, e.g. ``'state'`` or ``'control'``. By
            convention the declaration function name.
        component : Component or ComponentData
            The declared Pyomo component.
        **metadata
            Declaration details worth reading back, e.g. a control's
            ``profile``.
        """
        record = {"component": component}
        record.update(metadata)
        self._declarations.setdefault(kind, []).append(record)

    def record_transformation(self, name, **outcome):
        """Append ``name`` to the ordered transformation log.

        Parameters
        ----------
        name : str
            The transformation's registered name, e.g.
            ``'drto.dynamic_to_steady_state'``.
        **outcome
            Annotations of what the transformation did (what it freed or
            fixed, terms it dropped, whether it kept or collapsed the
            horizon), rendered in the registry view.
        """
        self._transformations.append({"name": name, "outcome": dict(outcome)})

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def declarations(self, kind=None):
        """Return declaration records, all of them or those of one kind.

        Parameters
        ----------
        kind : str, optional
            A declaration kind. When omitted, return a dict of every kind's
            record tuple.

        Returns
        -------
        tuple or dict
            The record dicts for ``kind`` (empty tuple if none), or the full
            ``{kind: records}`` mapping.
        """
        if kind is not None:
            return tuple(self._declarations.get(kind, ()))
        return {k: tuple(v) for k, v in self._declarations.items()}

    def components(self, kind):
        """Return the declared components of one kind, in declaration order.

        Parameters
        ----------
        kind : str
            A declaration kind.

        Returns
        -------
        tuple
            The declared components (empty if none).
        """
        return tuple(r["component"] for r in self._declarations.get(kind, ()))

    def has_declaration(self, kind):
        """Return whether any declaration of ``kind`` has been recorded."""
        return bool(self._declarations.get(kind))

    @property
    def transformations(self):
        """The ordered transformation log, a tuple of record dicts."""
        return tuple(self._transformations)

    def has_transformation(self, name):
        """Return whether the transformation ``name`` has been applied."""
        return any(r["name"] == name for r in self._transformations)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _role_lines(self):
        """Yield ``(label, text)`` pairs for the declaration view."""
        seen = set()
        ordered = [k for k, _ in _KIND_LABELS if k in self._declarations]
        extra = [k for k in self._declarations if k not in dict(_KIND_LABELS)]
        for kind in ordered + extra:
            seen.add(kind)
            label = dict(_KIND_LABELS).get(kind, kind)
            records = self._declarations[kind]
            ctype = _component_category(records[0]["component"])
            if ctype == "constraint":
                for r in records:
                    yield label, _compact_constraint(
                        r["component"], self._index_set_label(kind, r["component"])
                    )
            else:
                yield label, ", ".join(_entry(r, kind) for r in records)

    def _index_set_label(self, kind, con):
        """A display label for a kind's anonymous index set, or None.

        A stage cost is validated to be indexed over the samples minus the
        final time, so its nameless list-built set renders as the defining
        expression, ``sorted(t)[:-1]``. The members are checked against the
        recorded sample grid rather than assumed.
        """
        if kind not in ("tracking_stage_cost", "economic_stage_cost"):
            return None
        horizons = self.declarations("horizon")
        if not horizons:
            return None
        samples = horizons[0].get("samples")
        if samples is None or sorted(con.keys()) != list(samples[:-1]):
            return None
        return f"sorted({horizons[0]['component'].name})[:-1]"

    def _transformation_lines(self):
        """Yield one rendered line per applied transformation."""
        for r in self._transformations:
            notes = ", ".join(f"{k}={v}" for k, v in r["outcome"].items())
            yield f"{r['name']}" + (f": {notes}" if notes else "")

    def __repr__(self):
        lines = ["<drto registry>"]
        decls = list(self._role_lines())
        lines.append("declarations:" if decls else "declarations: (none)")
        for label, text in decls:
            lines.append(f"  {label}: {text}")
        tlines = list(self._transformation_lines())
        lines.append("transformations:" if tlines else "transformations: (none)")
        for t in tlines:
            lines.append(f"  {t}")
        return "\n".join(lines)

    def _repr_html_(self):
        rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(text)}"
            "</code></td></tr>"
            for label, text in self._role_lines()
        )
        titems = "".join(
            f"<li><code>{html.escape(t)}</code></li>"
            for t in self._transformation_lines()
        )
        return (
            "<div><b>drto registry</b>"
            f"<table><tbody>{rows}</tbody></table>"
            "<b>transformations</b>"
            + (f"<ol>{titems}</ol>" if titems else " (none)")
            + "</div>"
        )


def _component_category(comp):
    """Classify a declared component for rendering: constraint, set, or data."""
    ctype = getattr(comp, "ctype", None)
    name = getattr(ctype, "__name__", "")
    if name == "Constraint":
        return "constraint"
    if "Set" in name:
        return "set"
    return "data"


def _entry(record, kind):
    """Render one non-constraint declaration record as a short entry."""
    comp = record["component"]
    # only short scalar metadata renders; structural payloads (e.g. a
    # cost_group's terms) stay out of the view
    notes = [
        str(v)
        for k, v in record.items()
        if k != "component" and isinstance(v, (str, int, float, bool))
    ]
    of = record.get("of")
    if of is not None:  # a target pair renders with its owner
        notes.append(f"of {of.name}")
    if _component_category(comp) == "set":
        points = f"{len(comp)} points" if len(comp) else "no points"
        notes.append(f"{type(comp).__name__}, {points}")
    elif kind in ("state", "control"):
        notes.append(_var_status(comp))
    return comp.name + (f" ({', '.join(notes)})" if notes else "")


def _var_status(comp):
    """Return ``'free'``, ``'fixed'``, or a partial-fix count for a Var."""
    vals = list(comp.values()) if comp.is_indexed() else [comp]
    fixed = sum(1 for v in vals if v.fixed)
    if fixed == 0:
        return "free"
    if fixed == len(vals):
        return "fixed"
    return f"{fixed}/{len(vals)} fixed"


def _compact_constraint(con, index_set_label=None):
    """Render a constraint family as one symbolic equation.

    ``index_set_label`` overrides the displayed set of a single-index
    family whose set has no name of its own (the stage cost's sample
    list).

    Indexed constraints render with free indexes over their sets, named
    from the rule's own arguments, for example
    ``dzdt[t] == - z[t] + u[t]  for t in t``, via Pyomo's constraint
    templatization. Scalar constraints templatize too, which folds their
    internal sums over sets into symbolic ``SUM(...)`` form (the terminal
    cost of a large model stays readable). Rules that templatization cannot
    handle (for example ``Constraint.Skip`` guards) fall back to the raw
    expression for a scalar, or a representative member with its concrete
    index coordinates replaced by the free index names.
    """
    try:
        tmpl, indices = templatize_constraint(con)
        s = str(tmpl)
        names = _index_names(con, len(indices))
        for n in reversed(range(len(indices))):
            s = s.replace(f"_{n + 1}", names[n])
        if index_set_label is not None and len(indices) == 1:
            sets = [f"{names[0]} in {index_set_label}"]
        else:
            # each index of a multi-set constraint reports the whole
            # anonymous product as its set; when the counts line up,
            # position n of the product is index n's own set
            subsets = None
            if indices:
                prod = getattr(indices[0], "_set", None)
                if prod is not None:
                    cand = list(prod.subsets())
                    if len(cand) == len(indices):
                        subsets = cand
            sets = []
            for n, ix in enumerate(indices):
                iset = subsets[n] if subsets is not None else getattr(ix, "_set", None)
                sets.append(f"{names[n]} in {iset.name if iset is not None else '?'}")
        s = _name_sum_indices(s, names)
        return f"{s}  for {', '.join(sets)}" if sets else s
    except Exception:
        # Templatization executes the constraint rule on IndexTemplate
        # objects, so any rule logic (Skip guards comparing indices, dict
        # lookups, math on the index) can raise anything, and which exception
        # varies across Pyomo versions. Every failure means the same thing
        # here: this family cannot be templatized. Render a member with its
        # index coordinates replaced by the free index names, one per set.
        if not con.is_indexed():
            return str(con.expr)
        idx = next(iter(con.keys()))
        s = str(con[idx].expr)
        coords = idx if isinstance(idx, tuple) else (idx,)
        names = _index_names(con, len(coords))
        s = s.replace(
            "[" + ",".join(str(v) for v in coords) + "]", "[" + ",".join(names) + "]"
        )
        for n, (v, nm) in enumerate(zip(coords, names)):
            first = "[" if n == 0 else ","
            last = "]" if n == len(coords) - 1 else ","
            s = s.replace(f"{first}{v}{last}", f"{first}{nm}{last}")
        s = s.replace(f"[{coords[-1]}]", f"[{names[-1]}]")
        if index_set_label is not None and len(coords) == 1:
            return f"{s}  for {names[0]} in {index_set_label}"
        subsets = list(con.index_set().subsets())
        if len(subsets) == len(coords):
            tail = ", ".join(f"{nm} in {ss.name}" for nm, ss in zip(names, subsets))
        else:
            tail = f"{names[0]} in {con.index_set().name}"
        return f"{s}  for {tail}"
