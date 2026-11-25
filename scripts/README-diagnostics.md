# Offline diagnostics

This workflow is research-only and never connects to Supabase, Render, or the production forecast refresh.

Run locally with:
```bash
pip install -r requirements.txt pytest
pytest -q
python scripts/run_offline_diagnostics.py --download-sample --period 5y --output-dir artifacts
```

The sample stock defaults to `2330.TW` and is only a reproducible example, not a recommendation. Use `--ticker` to select another listed symbol.

The report includes:
- Annual regime distributions for the fixed current/candidate boundaries, trailing 252-session quantiles of historical three-session maximum moves, 20-session realized-volatility normalization, and 20-session true-range ATR normalization when adjusted OHLC is available.
- Strict historical router walk-forward D+1/D+2/D+3 metrics.
- Per-horizon stock holdout metrics.
- A fixed chronological holdout comparison across StandardScaler/no-scaling variants and logistic regularization strengths C = 0.01, 0.1, 1, 10, with raw probabilities, Platt Scaling, and Isotonic Regression. It reports Brier Score, Brier Skill versus the fold-specific prevalence baseline, Log Loss, and directional accuracy.

Calibration models are fitted only on matured development OOS predictions. Development labels whose forward horizon has not matured by the first holdout origin are excluded. The final chronological holdout is not used to fit calibrators, select C, choose scaling, or choose a calibration method. No manual probability-band clipping is applied; small numerical safeguards are used only for the Platt logit transform and metric stability.

The stock research workflow keeps the final chronological holdout out of model selection and calibration. Rolling alpha/beta features use trailing returns only. The historical router run omits a live inference timestamp, so all rows retain the strict one-session cross-market lag policy.

When OHLC is unavailable, the script reports the ATR method as unavailable and still runs close-to-close realized-volatility normalization; it does not substitute a fabricated ATR series.
