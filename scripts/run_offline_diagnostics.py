#!/usr/bin/env python3
"""Read-only historical diagnostics. Never connects to production storage."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from regime_router import (
    EXTREME_MIN_RETURN, LOW_MAX_RETURN, NORMAL_MAX_RETURN,
    build_regime_router_forecasts,
)
from stock_model_research import HORIZONS, _one_horizon, build_features

CURRENT = (LOW_MAX_RETURN, NORMAL_MAX_RETURN, EXTREME_MIN_RETURN)
CANDIDATE = (0.005, 0.012, 0.025)
TICKERS = {
    "TAIEX": "^TWII", "SOX": "^SOX", "TSM": "TSM", "VIX": "^VIX",
    "US10Y": "^TNX", "N225": "^N225", "KOSPI": "^KS11", "USDTWD": "TWD=X",
}


def download_close(ticker: str, period: str) -> pd.Series:
    import yfinance as yf
    frame = yf.download(ticker, period=period, auto_adjust=True,
                        progress=False, threads=False)
    if frame is None or frame.empty:
        raise RuntimeError(f"no rows returned for {ticker}")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    return close[~close.index.duplicated(keep="last")].sort_index()


def classify_custom(moves: pd.Series, limits: tuple[float, float, float]) -> pd.Series:
    low, normal, extreme = limits
    value = moves.abs()
    labels = pd.Series(pd.NA, index=moves.index, dtype="Int64")
    valid = value.notna()
    labels.loc[valid & (value <= low)] = 0
    labels.loc[valid & (value > low) & (value <= normal)] = 1
    labels.loc[valid & (value > normal) & (value < extreme)] = 2
    labels.loc[valid & (value >= extreme)] = 3
    return labels


def annual_regime_distribution(close: pd.Series) -> list[dict]:
    daily = close.pct_change(fill_method=None)
    future_rows = pd.concat(
        [daily.shift(-step).abs().rename(f"D+{step}") for step in (1, 2, 3)],
        axis=1,
    )
    # Exclude origins without all three future sessions; do not score partial horizons.
    future = future_rows.max(axis=1).where(future_rows.notna().all(axis=1))
    rows = []
    for label, limits in (("current", CURRENT), ("candidate", CANDIDATE)):
        classes = classify_custom(future, limits)
        frame = pd.DataFrame({"year": classes.index.year, "regime": classes}).dropna()
        for year, group in frame.groupby("year"):
            counts = group.regime.astype(int).value_counts()
            n = len(group)
            rows.append({
                "threshold_set": label, "year": int(year), "samples": int(n),
                **{f"{name}_n": int(counts.get(i, 0)) for i, name in enumerate(
                    ("low", "normal", "high", "extreme"))},
                **{f"{name}_pct": round(100 * counts.get(i, 0) / n, 2)
                   for i, name in enumerate(("low", "normal", "high", "extreme"))},
            })
    return rows


def make_router_features(market: pd.DataFrame) -> pd.DataFrame:
    ret = market.pct_change(fill_method=None)
    close = market["TAIEX"]
    x = pd.DataFrame(index=market.index)
    for name in ("SOX", "TSM", "VIX", "US10Y"):
        if name in ret:
            x[f"國際:{name}"] = ret[name]
    for name in ("N225", "KOSPI", "USDTWD"):
        if name in ret:
            x[f"區域:{name}"] = ret[name]
    x["本地:當日漲跌"] = ret["TAIEX"]
    x["本地:動能5日"] = close.pct_change(5, fill_method=None)
    x["本地:偏離20日均線"] = close / close.rolling(20, min_periods=15).mean() - 1
    x["本地:波動10日"] = ret["TAIEX"].rolling(10, min_periods=8).std()
    return x


def router_oos(market: pd.DataFrame) -> list[dict]:
    close = market["TAIEX"].dropna()
    market = market.reindex(close.index)
    x = make_router_features(market)
    inputs = {h: (x, close.shift(-h) / close - 1) for h in (1, 2, 3)}
    result = build_regime_router_forecasts(
        inputs, close, market_data=market, include_gate_sensitivity=False
    )  # No inference timestamp: strict historical lag policy.
    rows = []
    for item in result.get("sessions", []):
        m = item.get("routed_forecast_validation", {})
        rows.append({
            "horizon": f"D+{item.get('horizon_sessions')}",
            "status": item.get("status"),
            "samples": m.get("oos_samples", 0),
            "mae_points": m.get("routed_mae_points"),
            "direction_accuracy_pct": (
                round(100 * m["routed_direction_accuracy"], 2)
                if m.get("routed_direction_accuracy") is not None else None
            ),
            "bias_pp": m.get("mean_return_bias_pp"),
            "large_move_samples": m.get("large_move_samples", 0),
            "large_move_mae_points": m.get("large_move_routed_mae_points"),
            "large_move_direction_accuracy_pct": (
                round(100 * m["large_move_direction_accuracy"], 2)
                if m.get("large_move_direction_accuracy") is not None else None
            ),
            "large_move_bias_pp": m.get("large_move_mean_return_bias_pp"),
            "oos_start": m.get("oos_start_date"), "oos_end": m.get("oos_end_date"),
        })
    return rows


def stock_oos(stock: pd.Series, benchmark: pd.Series) -> dict:
    features = build_features(stock, benchmark)
    rows = {}
    for label, horizon in HORIZONS.items():
        result = _one_horizon(features, stock, horizon)
        rows[label] = {
            "status": result.get("status"),
            "selected_research_model": result.get("selected_research_model"),
            "training_samples": result.get("training_samples"),
            "holdout_start": result.get("holdout_start"),
            "holdout_end": result.get("holdout_end"),
            "untouched_holdout": result.get("untouched_holdout"),
            "annual_holdout_metrics": result.get("annual_holdout_metrics", {}),
            "candidate_comparison": result.get("candidate_comparison", {}),
            "note": result.get("note"),
        }
    return rows


def render(report: dict) -> str:
    lines = [
        "# Offline model diagnostics",
        f"- Generated UTC: {report['generated_at_utc']}",
        f"- Data source: {report['data_source']}",
        "- Scope: read-only historical research; no production DB writes.",
        "",
        "## Annual regime distribution (next 3 sessions' maximum absolute daily move)",
        "| Set | Year | N | Low % | Normal % | High % | Extreme % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["regime_distribution"]:
        lines.append(
            f"| {r['threshold_set']} | {r['year']} | {r['samples']} | "
            f"{r['low_pct']} | {r['normal_pct']} | {r['high_pct']} | {r['extreme_pct']} |"
        )
    lines += [
        "", "## Router walk-forward OOS",
        "| Horizon | Status | N | MAE pts | Direction % | Bias pp | Large N | Large MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["router_oos"]:
        lines.append(
            f"| {r['horizon']} | {r['status']} | {r['samples']} | {r['mae_points']} | "
            f"{r['direction_accuracy_pct']} | {r['bias_pp']} | "
            f"{r['large_move_samples']} | {r['large_move_mae_points']} |"
        )
    lines += ["", "## Stock research holdout"]
    for label, r in report["stock_oos"].items():
        lines += [
            "", f"### {label}",
            f"- Status/model: {r.get('status')} / {r.get('selected_research_model')}",
            f"- Holdout: {json.dumps(r.get('untouched_holdout'), ensure_ascii=False)}",
            f"- Annual OOS: {json.dumps(r.get('annual_holdout_metrics', {}), ensure_ascii=False)}",
        ]
    if report["warnings"]:
        lines += ["", "## Warnings"] + [f"- {warning}" for warning in report["warnings"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--ticker", default="2330.TW")
    parser.add_argument("--period", default="10y")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_source": "Yahoo Finance read-only sample" if args.download_sample else "no data source configured",
        "thresholds": {
            "current": {"low_max": CURRENT[0], "normal_max": CURRENT[1], "extreme_min": CURRENT[2]},
            "candidate": {"low_max": CANDIDATE[0], "normal_max": CANDIDATE[1], "extreme_min": CANDIDATE[2]},
        },
        "regime_distribution": [], "router_oos": [], "stock_oos": {}, "warnings": [],
    }
    if not args.download_sample:
        report["warnings"].append("Pass --download-sample to fetch a public historical sample for CI diagnostics.")
    else:
        downloaded = {}
        for name, ticker in TICKERS.items():
            try:
                downloaded[name] = download_close(ticker, args.period)
            except Exception as exc:
                report["warnings"].append(f"{ticker}: {type(exc).__name__}: {exc}")
        market = pd.DataFrame(downloaded).sort_index() if "TAIEX" in downloaded else pd.DataFrame()
        if len(market) and market["TAIEX"].notna().sum() > 500:
            report["regime_distribution"] = annual_regime_distribution(market["TAIEX"].dropna())
            try:
                report["router_oos"] = router_oos(market)
            except Exception as exc:
                report["warnings"].append(f"Router OOS failed: {type(exc).__name__}: {exc}")
        else:
            report["warnings"].append("TAIEX price history unavailable; regime/router metrics skipped.")
        try:
            stock = download_close(args.ticker, args.period)
            benchmark = download_close("^TWII", args.period)
            aligned = pd.concat([stock.rename("stock"), benchmark.rename("market")], axis=1).dropna()
            if len(aligned) > 800:
                report["stock_ticker"] = args.ticker
                report["stock_oos"] = stock_oos(aligned["stock"], aligned["market"])
            else:
                report["warnings"].append("Stock/benchmark history is shorter than 800 aligned sessions.")
        except Exception as exc:
            report["warnings"].append(f"Stock OOS skipped: {type(exc).__name__}: {exc}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "offline_diagnostics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "offline_diagnostics.md").write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
