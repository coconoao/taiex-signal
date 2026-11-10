"""Run a point-in-time regime-router backtest on real Yahoo market history.

This research job is read-only with respect to production storage. It downloads
price history into the runner's temporary filesystem and does not call the
pipeline's production persistence path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure pipeline.load_live() does not persist caches to Supabase.
if "--skip-backfill" not in sys.argv:
    sys.argv.append("--skip-backfill")

import pipeline  # noqa: E402
from regime_router import build_regime_router_forecasts  # noqa: E402


def main() -> int:
    cache_path = "/tmp/taiex-regime-router-market-history.csv"
    market = pipeline.load_live(years=10, cache_path=cache_path)
    market = market.sort_index()
    close = market["TAIEX"].dropna()
    if len(close) < 800:
        raise RuntimeError(
            f"真實台股歷史資料不足：{len(close)} 個交易日，至少需要 800 日"
        )

    features_by_horizon = {}
    for horizon in (1, 2, 3):
        X, _, forward_returns = pipeline.build(
            market, chips=None, margin=None, futures=None, h=horizon
        )
        features_by_horizon[horizon] = (X, forward_returns)

    result = build_regime_router_forecasts(
        features_by_horizon,
        close,
        target_meta={},
        market_data=market,
    )
    result["real_data_audit"] = {
        "source": "Yahoo Finance daily close with TWSE recent official TAIEX refresh",
        "start_date": str(close.index.min().date()),
        "end_date": str(close.index.max().date()),
        "taiwan_sessions": int(len(close)),
        "features": [
            "TAIEX",
            "SOX",
            "TSM",
            "VIX",
            "US10Y",
            "N225",
            "KOSPI",
            "USDTWD",
        ],
        "optional_layers": {
            "institutional_flows": False,
            "margin_short_selling": False,
            "futures_open_interest": False,
        },
        "timing_policy": result.get("feature_timing_policy"),
        "note": (
            "價格層真實歷史樣本外研究；不含法人、融資融券及期貨層，"
            "不可視為完整正式模型驗證。所有跨市場特徵依台股收盤可得時間落後一個台灣交易日。"
        ),
    }

    for session in result.get("sessions", []):
        validation = session.get("router_validation", {})
        forecast = session.get("routed_forecast_validation", {})
        print(
            "REAL_DATA D+{} status={} regime={} expert={} OOS={} "
            "macro_f1={} high_extreme_recall={} high_extreme_precision={} "
            "routed_mae={} global_ridge_mae={} max_move_mae={} global_max_move_mae={}".format(
                session.get("horizon_sessions"),
                session.get("status"),
                session.get("regime"),
                session.get("expert_used"),
                validation.get("oos_samples"),
                validation.get("macro_f1"),
                validation.get("high_extreme_recall"),
                validation.get("high_extreme_precision"),
                forecast.get("routed_mae_points"),
                forecast.get("global_ridge_mae_points"),
                forecast.get("max_daily_move_mae_pct"),
                forecast.get("global_max_daily_move_mae_pct"),
            ),
            flush=True,
        )

    output = Path("regime-router-real-data.json")
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    if len(result.get("sessions", [])) != 3:
        raise RuntimeError("真實資料回測未產生完整 D+1、D+2、D+3 結果")
    if any(
        item.get("router_validation", {}).get("status") != "available"
        or item.get("routed_forecast_validation", {}).get("status") != "available"
        for item in result["sessions"]
    ):
        raise RuntimeError("真實資料樣本外驗證不足；已輸出診斷 JSON，不能宣稱模型通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
