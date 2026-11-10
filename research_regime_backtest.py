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


def _restore_supabase_research_cache(cache_dir: Path) -> dict:
    """Read only the four historical CSV artifacts needed for point-in-time research."""
    import urllib.parse
    import storage

    names = ("market_history.csv", "chips.csv", "margin.csv", "futures.csv")
    if not storage.enabled():
        print("Supabase read-only cache unavailable; running price-only diagnostic.", flush=True)
        return {"enabled": False, "restored": []}
    query = urllib.parse.urlencode({
        "select": "key,content",
        "key": "in.(" + ",".join(names) + ")",
    })
    rows = storage._request("GET", storage.TABLE + "?" + query)
    restored = []
    for row in rows:
        name, content = row.get("key"), row.get("content")
        if name in names and isinstance(content, str):
            (cache_dir / name).write_text(content, encoding="utf-8")
            restored.append(name)
    missing = sorted(set(names) - set(restored))
    if missing:
        raise RuntimeError("Supabase 歷史研究快取缺少必要檔案：" + ", ".join(missing))
    print("已唯讀取得 Supabase 歷史快取：" + ", ".join(restored), flush=True)
    return {"enabled": True, "restored": restored}


def _read_optional_frame(cache_dir: Path, name: str):
    path = cache_dir / name
    if not path.is_file() or path.stat().st_size == 0:
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _coverage(frame):
    if frame is None:
        return {"available": False, "rows": 0, "start_date": None, "end_date": None}
    valid = frame.dropna(how="all")
    return {
        "available": bool(len(valid)),
        "rows": int(len(valid)),
        "start_date": str(valid.index.min().date()) if len(valid) else None,
        "end_date": str(valid.index.max().date()) if len(valid) else None,
    }


def main() -> int:
    cache_dir = Path("/tmp/taiex-regime-router-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_status = _restore_supabase_research_cache(cache_dir)
    cache_path = str(cache_dir / "market_history.csv")
    market = pipeline.load_live(years=10, cache_path=cache_path)
    market = market.sort_index()
    close = market["TAIEX"].dropna()
    if len(close) < 800:
        raise RuntimeError(
            f"真實台股歷史資料不足：{len(close)} 個交易日，至少需要 800 日"
        )

    chips_raw = _read_optional_frame(cache_dir, "chips.csv")
    margin_raw = _read_optional_frame(cache_dir, "margin.csv")
    futures_raw = _read_optional_frame(cache_dir, "futures.csv")
    # Shift raw source frames before pipeline feature engineering. This is
    # essential for margin: pipeline.build() forward-fills delayed publications,
    # so shifting the finished features would accidentally add a second lag.
    chips = __import__("regime_router")._lag_post_close_source_frame(chips_raw, market.index)
    margin = __import__("regime_router")._lag_post_close_source_frame(margin_raw, market.index)
    futures = __import__("regime_router")._lag_post_close_source_frame(futures_raw, market.index)

    features_by_horizon = {}
    for horizon in (1, 2, 3):
        X, _, forward_returns = pipeline.build(
            market, chips=chips, margin=margin, futures=futures, h=horizon
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
            "TAIEX", "SOX", "TSM", "VIX", "US10Y", "N225", "KOSPI", "USDTWD",
            *[str(column) for column in features_by_horizon[1][0].columns
              if str(column).startswith(("籌碼:", "信用:", "期貨:"))],
        ],
        "cache_source": "Supabase site_data (read-only)" if cache_status["enabled"] else "unavailable",
        "optional_layers": {
            "institutional_flows": bool(features_by_horizon[1][0].columns.astype(str).str.startswith("籌碼:").any()),
            "margin_short_selling": bool(features_by_horizon[1][0].columns.astype(str).str.startswith("信用:").any()),
            "futures_open_interest": bool(features_by_horizon[1][0].columns.astype(str).str.startswith("期貨:").any()),
        },
        "optional_source_coverage": {
            "institutional_flows": _coverage(chips_raw),
            "margin_short_selling": _coverage(margin_raw),
            "futures_open_interest": _coverage(futures_raw),
        },
        "timing_policy": result.get("feature_timing_policy"),
        "note": (
            "使用 Supabase 唯讀歷史快取；法人、融資融券及期貨原始資料先依完整台灣交易日曆延後一個交易日，"
            "再進行特徵工程。跨市場收盤特徵亦延後一個台灣交易日。若資料來源不可用則明確標示，不以未公布資料回填。"
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
