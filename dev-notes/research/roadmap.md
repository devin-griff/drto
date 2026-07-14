# Roadmap

Status: design complete, no code. Seeded from DESIGN.md (Delivery plan and
Follow-on). DESIGN.md stays authoritative for decisions; this note tracks
sequence.

## Near term

1. Quad-tank spike: the ideal-NMPC mode end-to-end on the quad tank (build on
   the Quad_tank_cvp model), then flip on the advanced-step mode with the
   existing pounce `estimate()`. The three-variant comparison plot falls out.
2. v1 package: the loop, the three execution variants, the control-side
   declarations, tests, docs, executed notebooks.

## Follow-on

- Moving horizon estimation (the estimation half): the six estimation
  declarations, the soft arrival cost, and covariance propagation for the
  arrival-cost weight (pounce covariance machinery).
- Steady-state reduction wiring (setpoint consistency and economic RTO) via
  the reusable-object mechanism.

## Stretch

- One dynamic IDAES flowsheet as a credibility example.
