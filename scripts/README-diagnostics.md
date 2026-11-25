# Offline diagnostics

This workflow is research-only and never connects to Supabase, Render, or the production forecast refresh.

Run locally with:
```bash
pip install -r requirements.txt pytest
pytest -q
python scripts/run_offline_diagnostics.py --download-sample --period 10y --output-dir artifacts
```

The sample stock defaults to `2330.TW` and is only a reproducible example, not a recommendation. Use `--ticker` to select another listed symbol. The report records annual regime distributions for the current and candidate boundaries, walk-forward D+1/D+2/D+3 router metrics, and per-horizon stock research holdout metrics. Missing public data is recorded as a warning rather than silently replaced with synthetic observations.

The stock research workflow keeps the final chronological holdout out of model selection and calibration. Rolling alpha/beta features use trailing returns only. The historical router run omits a live inference timestamp, so all rows retain the strict one-session cross-market lag policy.
