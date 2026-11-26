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
from stock_model_research import HORIZONS, _one_horizon, build_features, prequential_platt_candidates
from scripts.final_oos_methods import atr_percent
from research_experiments import (annual_regime_experiments, probability_calibration_experiment)

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

def download_ohlc(ticker: str, period: str) -> pd.DataFrame:
    """Download adjusted OHLC needed for a true-range ATR diagnostic."""
    import yfinance as yf
    frame = yf.download(ticker, period=period, auto_adjust=True,
                        progress=False, threads=False)
    if frame is None or frame.empty:
        raise RuntimeError(f"no OHLC rows returned for {ticker}")

    def column(name):
        values = frame[name]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        return pd.to_numeric(values, errors="coerce")

    out = pd.DataFrame({"high": column("High"), "low": column("Low"),
                        "close": column("Close")}).dropna(subset=["high", "low", "close"])
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()



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


def annual_regime_distribution(close, high=None, low=None):
    """Compare fixed and volatility-adaptive annual regime distributions."""
    return annual_regime_experiments(
        close, high=high, low=low, current=CURRENT, candidate=CANDIDATE
    )


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


def router_oos(market: pd.DataFrame, ohlc: pd.DataFrame | None = None) -> dict:
    """Run fixed and ATR router refits on identical features and OOS origins."""
    close = market["TAIEX"].dropna()
    aligned_market = market.reindex(close.index)
    x = make_router_features(aligned_market)
    inputs = {h: (x, close.shift(-h) / close - 1) for h in (1, 2, 3)}
    atr = None
    if ohlc is not None:
        atr = atr_percent(
            ohlc["high"].reindex(close.index),
            ohlc["low"].reindex(close.index),
            ohlc["close"].reindex(close.index),
            window=20,
        )
    results = {}
    for mode in ("fixed", "atr"):
        if mode == "atr" and atr is None:
            results[mode] = {"status": "unavailable", "reason": "OHLC data unavailable"}
            continue
        result = build_regime_router_forecasts(
            inputs, close, market_data=aligned_market,
            include_gate_sensitivity=False,
            regime_threshold_mode=mode, atr_pct=atr,
            evaluation_start_date="2024-01-01",
        )
        rows = []
        for item in result.get("sessions", []):
            metrics = item.get("routed_forecast_validation", {})
            annual = item.get("annual_oos_metrics", {})
            rows.append({
                "horizon": f"D+{item.get('horizon_sessions')}",
                "status": item.get("status"),
                "samples": metrics.get("oos_samples", 0),
                "oos_start": metrics.get("oos_start_date"),
                "oos_end": metrics.get("oos_end_date"),
                "oos_dates": item.get("oos_forecast_dates", []),
                "mae_points": metrics.get("routed_mae_points"),
                "direction_accuracy_pct": (
                    round(100 * metrics["routed_direction_accuracy"], 2)
                    if metrics.get("routed_direction_accuracy") is not None else None
                ),
                "bias_pp": metrics.get("mean_return_bias_pp"),
                "non_normal_regime_pct": (
                    round(100.0 * sum(v.get("samples", 0) for k, v in metrics.get("by_actual_regime", {}).items() if k != "normal") / metrics["oos_samples"], 4)
                    if metrics.get("oos_samples") else None
                ),
                "signal_non_normal_regime_pct": item.get("signal_non_normal_regime_pct"),
                "atr_warmup_fallback_samples": item.get("atr_warmup_fallback_samples", 0),
                "router_validation": item.get("router_validation", {}),
                "annual": annual,
            })
        results[mode] = {"status": result.get("status"), "horizons": rows}
    comparison = []
    fixed_rows = {x["horizon"]: x for x in results.get("fixed", {}).get("horizons", [])}
    atr_rows = {x["horizon"]: x for x in results.get("atr", {}).get("horizons", [])}
    for horizon in sorted(set(fixed_rows) & set(atr_rows)):
        f, a = fixed_rows[horizon], atr_rows[horizon]
        comparison.append({
            "horizon": horizon,
            "same_oos_dates": f.get("oos_dates") == a.get("oos_dates"),
            "fixed_samples": f.get("samples"), "atr_samples": a.get("samples"),
            "fixed_start": f.get("oos_start"), "fixed_end": f.get("oos_end"),
            "atr_start": a.get("oos_start"), "atr_end": a.get("oos_end"),
            "mae_delta_atr_minus_fixed": (
                a["mae_points"] - f["mae_points"]
                if a.get("mae_points") is not None and f.get("mae_points") is not None else None
            ),
            "direction_delta_pp_atr_minus_fixed": (
                a["direction_accuracy_pct"] - f["direction_accuracy_pct"]
                if a.get("direction_accuracy_pct") is not None and f.get("direction_accuracy_pct") is not None else None
            ),
        })
    return {
        "data_start": str(close.index.min().date()) if len(close) else None,
        "data_end": str(close.index.max().date()) if len(close) else None,
        "atr_warmup_sessions": 20,
        "atr_warmup_fallback_total_sessions": int(atr.isna().sum()) if atr is not None else None,
        "atr_threshold_ratios": [0.5, 1.0, 2.0],
        "by_mode": results,
        "paired_comparison": comparison,
    }

def stock_oos(stock: pd.Series, benchmark: pd.Series) -> tuple[dict, dict, dict]:
    features = build_features(stock, benchmark)
    rows, calibration = {}, {}
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
        calibration[label] = probability_calibration_experiment(features, stock, horizon)
    prequential = prequential_platt_candidates(features, stock, horizon=21, oos_start="2024-01-01")
    return rows, calibration, prequential


def render(report: dict) -> str:
    lines = [
        "# Offline model diagnostics",
        f"- Generated UTC: {report['generated_at_utc']}",
        f"- Data source: {report['data_source']}",
        "- Scope: read-only historical research; no production DB writes.",
        "",
        "## Annual regime distribution",
        "| Method | Year | N | Low % | Normal % | High % | Extreme % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report.get("regime_distribution", []):
        lines.append(
            f"| {r['threshold_set']} | {r['year']} | {r['samples']} | "
            f"{r['low_pct']} | {r['normal_pct']} | {r['high_pct']} | {r['extreme_pct']} |"
        )

    router = report.get("router_oos", {})
    lines += [
        "", "## Paired router OOS: fixed versus 20-session ATR%",
        f"- Full price data range: {router.get('data_start')} to {router.get('data_end')}",
        "- OOS evaluation starts 2024-01-01; both variants use the same features, training origins, and test origins.",
        "- ATR: trailing 20-session simple true-range average / close; missing or warmup ATR falls back to fixed thresholds.",
        "- Signal frequency means the share of OOS forecasts routed to a non-normal regime.",
        f"- ATR unavailable/warmup rows in full input history: {router.get('atr_warmup_fallback_total_sessions')}",
        "| Mode | Horizon | N | OOS start | OOS end | MAE pts | Direction % | Multiclass Brier | Log Loss | Signal frequency % | OOS warmup fallback N |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("fixed", "atr"):
        for row in router.get("by_mode", {}).get(mode, {}).get("horizons", []):
            rv = row.get("router_validation", {})
            lines.append(
                f"| {mode} | {row['horizon']} | {row['samples']} | {row['oos_start']} | {row['oos_end']} | "
                f"{row['mae_points']} | {row['direction_accuracy_pct']} | {rv.get('multiclass_brier')} | "
                f"{rv.get('log_loss')} | {row.get('signal_non_normal_regime_pct')} | "
                f"{row.get('atr_warmup_fallback_samples')} |"
            )

    lines += [
        "", "### Overall paired comparison",
        "| Horizon | Exact same OOS dates | Fixed N | ATR N | MAE delta (ATR-fixed) | Direction delta pp | Multiclass Brier delta | Log Loss delta | Signal frequency multiple |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in router.get("paired_comparison", []):
        fixed_row = next((x for x in router.get("by_mode", {}).get("fixed", {}).get("horizons", []) if x["horizon"] == pair["horizon"]), {})
        atr_row = next((x for x in router.get("by_mode", {}).get("atr", {}).get("horizons", []) if x["horizon"] == pair["horizon"]), {})
        fval, aval = fixed_row.get("router_validation", {}), atr_row.get("router_validation", {})
        fp, ap = fixed_row.get("signal_non_normal_regime_pct"), atr_row.get("signal_non_normal_regime_pct")
        multiple = round(ap / fp, 4) if fp not in (None, 0) and ap is not None else None
        lines.append(
            f"| {pair['horizon']} | {pair['same_oos_dates']} | {pair['fixed_samples']} | {pair['atr_samples']} | "
            f"{pair['mae_delta_atr_minus_fixed']} | {pair['direction_delta_pp_atr_minus_fixed']} | "
            f"{(aval.get('multiclass_brier') - fval.get('multiclass_brier')) if aval.get('multiclass_brier') is not None and fval.get('multiclass_brier') is not None else None} | "
            f"{(aval.get('log_loss') - fval.get('log_loss')) if aval.get('log_loss') is not None and fval.get('log_loss') is not None else None} | {multiple} |"
        )

    lines += [
        "", "### Annual paired OOS metrics",
        "| Year | Horizon | Fixed N | ATR N | Fixed dates | ATR dates | Fixed MAE | ATR MAE | Fixed direction % | ATR direction % | Fixed Brier | ATR Brier | Fixed LogLoss | ATR LogLoss | Signal frequency multiple | ATR warmup N |",
        "|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in sorted({x["horizon"] for x in router.get("by_mode", {}).get("fixed", {}).get("horizons", [])}):
        fixed_row = next((x for x in router.get("by_mode", {}).get("fixed", {}).get("horizons", []) if x["horizon"] == horizon), {})
        atr_row = next((x for x in router.get("by_mode", {}).get("atr", {}).get("horizons", []) if x["horizon"] == horizon), {})
        years = sorted(set(fixed_row.get("annual", {})) | set(atr_row.get("annual", {})))
        for year in years:
            f, a = fixed_row.get("annual", {}).get(year, {}), atr_row.get("annual", {}).get(year, {})
            fv, av = f.get("router_validation", {}), a.get("router_validation", {})
            fp, ap = f.get("non_normal_regime_pct"), a.get("non_normal_regime_pct")
            multiple = round(ap / fp, 4) if fp not in (None, 0) and ap is not None else None
            fd = f"{f.get('oos_start_date')}–{f.get('oos_end_date')}"
            ad = f"{a.get('oos_start_date')}–{a.get('oos_end_date')}"
            lines.append(
                f"| {year} | {horizon} | {f.get('oos_samples')} | {a.get('oos_samples')} | {fd} | {ad} | "
                f"{f.get('routed_mae_points')} | {a.get('routed_mae_points')} | "
                f"{100 * f.get('routed_direction_accuracy') if f.get('routed_direction_accuracy') is not None else None} | "
                f"{100 * a.get('routed_direction_accuracy') if a.get('routed_direction_accuracy') is not None else None} | "
                f"{fv.get('multiclass_brier')} | {av.get('multiclass_brier')} | {fv.get('log_loss')} | {av.get('log_loss')} | "
                f"{multiple} | {a.get('atr_warmup_fallback_samples')} |"
            )

    lines += ["", "## Stock research holdout"]
    for label, row in report.get("stock_oos", {}).items():
        lines += [
            "", f"### {label}",
            f"- Status/model: {row.get('status')} / {row.get('selected_research_model')}",
            f"- Holdout: {json.dumps(row.get('untouched_holdout'), ensure_ascii=False)}",
            f"- Annual OOS: {json.dumps(row.get('annual_holdout_metrics', {}), ensure_ascii=False)}",
        ]

    lines += [
        "", "## Probability calibration and regularization experiments",
        "Existing chronological holdout experiments; calibrators use matured development OOS only.",
        "| Horizon | Scaling | C | Method | N | Brier | Brier Skill | Log Loss | Direction % |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for horizon, experiment in report.get("probability_calibration", {}).items():
        for config in experiment.get("experiments", []):
            for method, result in config.get("methods", {}).items():
                metrics = result.get("holdout_metrics")
                if result.get("status") == "available" and metrics:
                    lines.append(
                        f"| {horizon} | {config['scaling']} | {config['C']} | {method} | "
                        f"{metrics['samples']} | {metrics['brier_score']} | {metrics['brier_skill']} | "
                        f"{metrics['log_loss']} | {metrics['direction_accuracy_pct']} |"
                    )

    preq = report.get("prequential_platt", {})
    lines += [
        "", "## Expanding-window prequential Platt (21-session purge)",
        f"- Actual OOS range: {preq.get('oos_start_actual')} to {preq.get('oos_end_actual')}",
        "- Base models: unscaled LogisticRegression C=0.1 and C=0.01; Platt C=1.0.",
        "- Calibration fit uses only prior matured OOS labels, purged against the complete trading-session index.",
        "| Base C | Year | N | Brier | Brier Skill | Brier Skill 95% CI | Log Loss | LogLoss delta 95% CI | Direction % | Baseline direction % | Delta pp | All slope | All intercept | All ECE | Fitted N | Fitted slope | Fitted intercept | Fitted ECE | Coverage % |",
        "|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate, result in preq.get("candidates", {}).items():
        if result.get("status") != "available":
            lines.append(f"| {candidate} | all | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | 0 |")
            continue
        for year, metrics in [("all", result.get("overall", {})), *result.get("annual", {}).items()]:
            lines.append(
                f"| {candidate} | {year} | {metrics.get('samples')} | {metrics.get('brier_score')} | "
                f"{metrics.get('brier_skill')} | {metrics.get('brier_skill_95pct_block_bootstrap_ci')} | "
                f"{metrics.get('log_loss')} | {metrics.get('log_loss_delta_95pct_block_bootstrap_ci')} | "
                f"{metrics.get('direction_accuracy_pct')} | {metrics.get('baseline_direction_accuracy_pct')} | "
                f"{metrics.get('direction_delta_vs_baseline_pp')} | {metrics.get('calibration_slope')} | "
                f"{metrics.get('calibration_intercept')} | {metrics.get('ece_10_bins')} | "
                f"{metrics.get('fitted_only_samples')} | {metrics.get('fitted_only_calibration_slope')} | "
                f"{metrics.get('fitted_only_calibration_intercept')} | {metrics.get('fitted_only_ece_10_bins')} | "
                f"{metrics.get('calibration_coverage_pct')} |"
            )
    if report.get("warnings"):
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
        "regime_distribution": [], "router_oos": {}, "stock_oos": {},
        "probability_calibration": {}, "prequential_platt": {}, "warnings": [],
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
            taiwan_ohlc = None
            try:
                taiwan_ohlc = download_ohlc("^TWII", args.period)
            except Exception as exc:
                report["warnings"].append(f"TAIEX OHLC/ATR unavailable: {type(exc).__name__}: {exc}")
            report["regime_distribution"] = annual_regime_distribution(
                market["TAIEX"].dropna(),
                high=taiwan_ohlc["high"] if taiwan_ohlc is not None else None,
                low=taiwan_ohlc["low"] if taiwan_ohlc is not None else None,
            )
            try:
                report["router_oos"] = router_oos(market, taiwan_ohlc)
            except Exception as exc:
                report["warnings"].append(f"Router OOS failed: {type(exc).__name__}: {exc}")
        else:
            report["warnings"].append("TAIEX price history unavailable; regime/router metrics skipped.")
        try:
            stock = download_close(args.ticker, args.period)
            benchmark = download_close("^TWII", args.period)
            aligned = pd.concat([stock.rename("stock"), benchmark.rename("market")], axis=1, sort=False).dropna()
            if len(aligned) > 800:
                report["stock_ticker"] = args.ticker
                report["stock_oos"], report["probability_calibration"], report["prequential_platt"] = stock_oos(
                    aligned["stock"], aligned["market"]
                )
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
