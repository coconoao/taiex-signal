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

At this stage the dynamic labels are diagnostic targets only. They do not yet
train or replace the production router. A dynamic-label classifier must be
trained and evaluated walk-forward against both target definitions before any
claim that dynamic regimes improve routing.

## Prediction intervals

The module compares two 50%, 80%, and 90% interval constructions around
point-in-time OOS predictions:

- Symmetric absolute-residual conformal intervals use a finite-sample quantile
  of absolute errors from earlier OOS forecasts.
- Asymmetric signed-residual intervals use separate finite-sample lower and
  upper quantiles of signed errors, allowing forecast-error skew.

Both variants use only prior OOS residuals whose targets have matured
(prior origin + horizon <= current origin). The caller invokes the procedure
separately for each horizon. The asymmetric variant is a research comparator,
not an automatic replacement: compare coverage, mean width, and interval score
on identical OOS origins. Do not select a method solely because its intervals
are narrower.

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
The calendar parser fails closed on a year mismatch, malformed or duplicate
dates, inconsistent declared row counts, too few records or weekday closures,
missing key annual holiday periods, and explicitly published weekend trading
that the weekday-only session builder cannot represent. Q3 may legitimately
have no closure in years when the Mid-Autumn holiday falls in Q4.
