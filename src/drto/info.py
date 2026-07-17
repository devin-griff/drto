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

from pyomo.core.expr.template_expr import templatize_constraint

#: The ``Block.private_data`` scope drto stores its registry under.
_DRTO_SCOPE = "drto"

#: Display order and labels for the declaration kinds of the control-side
#: surface (feature 002). Unknown kinds render after these, labeled by their
#: raw kind string, so the registry does not need updating for new kinds.
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
)

#: Free-index names used when rendering indexed constraints symbolically.
_INDEX_NAMES = ("k", "j", "i", "l")


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
                    yield label, _compact_constraint(r["component"])
            else:
                yield label, ", ".join(_entry(r, kind) for r in records)

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
        rows = "".join(f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(text)}" "</code></td></tr>" for label, text in self._role_lines())
        titems = "".join(f"<li><code>{html.escape(t)}</code></li>" for t in self._transformation_lines())
        return "<div><b>drto registry</b>" f"<table><tbody>{rows}</tbody></table>" "<b>transformations</b>" + (f"<ol>{titems}</ol>" if titems else " (none)") + "</div>"


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
    notes = [str(v) for k, v in record.items() if k != "component" and isinstance(v, (str, int, float, bool))]
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


def _compact_constraint(con):
    """Render a constraint family as one symbolic equation.

    Indexed constraints render with a free index over their set, for example
    ``dzdt[k] == - z[k] + u[k]  for k in t``, via Pyomo's constraint
    templatization. Rules that templatization cannot handle (for example
    ``Constraint.Skip`` guards) fall back to a representative member with its
    concrete index replaced by the free index.
    """
    if not con.is_indexed():
        return str(con.expr)
    try:
        tmpl, indices = templatize_constraint(con)
        s = str(tmpl)
        sets = []
        for n in reversed(range(len(indices))):
            name = _INDEX_NAMES[n % len(_INDEX_NAMES)]
            s = s.replace(f"_{n + 1}", name)
        for n, ix in enumerate(indices):
            name = _INDEX_NAMES[n % len(_INDEX_NAMES)]
            iset = getattr(ix, "_set", None)
            sets.append(f"{name} in {iset.name if iset is not None else '?'}")
        return f"{s}  for {', '.join(sets)}"
    except Exception:
        # Templatization executes the constraint rule on IndexTemplate
        # objects, so any rule logic (Skip guards comparing indices, dict
        # lookups, math on the index) can raise anything, and which exception
        # varies across Pyomo versions. Every failure means the same thing
        # here: this family cannot be templatized, show a representative.
        idx = next(iter(con.keys()))
        s = str(con[idx].expr)
        for frm, to in ((f"[{idx}]", "[k]"), (f"[{idx},", "[k,")):
            s = s.replace(frm, to)
        return f"{s}  for k in {con.index_set().name} (shown at {idx})"
