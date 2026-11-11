# Dynamic regime and prediction-interval research v1

This branch adds research utilities only. It does not modify production forecasts,
the formal Ridge baseline, model weights, Supabase data, or dashboard rendering.

## Dynamic four-regime labels

For each horizon (D+1, D+2, D+3) independently, row t is labelled using
expanding quantiles of strictly prior realized maximum absolute one-session
moves. Candidate cut points are Q25 / Q75 / Q95:
- low: value <= prior Q25
- normal: Q25 < value <= prior Q75
- high: Q75 < value <= prior Q95
- extreme: value > prior Q95

A 252-observation warm-up is the default research target. Before enough matured
history exists, labels and thresholds remain unavailable. These quantiles are
candidates, not approved production thresholds. Class counts and extreme-class
support must be reported before interpretation.

## Prediction intervals

The module creates symmetric 50%, 80%, and 90% intervals around supplied
point-in-time OOS predictions. Radius is a finite-sample conformal quantile of
absolute residuals from earlier OOS forecasts only. A residual is eligible only
after its target has matured (prior origin + horizon <= current origin).
The caller must invoke the procedure separately for each horizon.

Regime-conditional calibration is optional. It is used only after the prior
matured sample count in that regime reaches the configured minimum; otherwise
the interval falls back to the same-horizon global residual pool and marks the
fallback explicitly. It never borrows another horizon's residuals.

## Required validation before any production use

1. Use official Taiwan session positions; do not infer missing sessions from observed prices.
2. Use predictions generated walk-forward, with purging for each horizon.
3. Report empirical coverage, mean width, and interval score for every nominal level, horizon, and regime; include calibration sample counts.
4. Compare dynamic labels with fixed-threshold labels on identical OOS origins.
5. Report class prevalence, confusion matrix, macro-F1, high/extreme recall, and extreme sample support.
6. Keep Ridge unchanged. Do not promote any regime expert unless mature OOS error beats Ridge under a predeclared gate.
7. Treat coverage shortfall, unavailable calendar, or inadequate sample count as failed or inconclusive validation, not a pass.

The research backtest now optionally emits these diagnostics for the primary
price-only OOS run. They are not wired to the production dashboard or production
forecast path. Real-market interpretation remains blocked until an independent
official Taiwan trading-session calendar is available and passes completeness
validation; observed-price dates must not be treated as a substitute calendar.
