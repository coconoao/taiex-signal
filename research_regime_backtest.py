"""Run a point-in-time regime-router backtest on real Yahoo market history.

This research job is read-only with respect to production storage. It downloads
price history into the runner's temporary filesystem and does not call the
pipeline's production persistence path.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd
from pathlib import Path

# Ensure pipeline.load_live() does not persist caches to Supabase.
if "--skip-backfill" not in sys.argv:
    sys.argv.append("--skip-backfill")

import pipeline  # noqa: E402
from regime_router import build_regime_router_forecasts, _lag_post_close_source_frame  # noqa: E402


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
    if os.environ.get("REGIME_ROUTER_REQUIRE_SUPABASE_CACHE") == "1" and not cache_status["enabled"]:
        raise RuntimeError("本次真實資料驗證要求 Supabase 唯讀歷史快取，但連線設定不可用")
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
    # Shift raw source frames before feature engineering. The full Taiwan
    # trading calendar is used so missing source dates cannot compress the lag.
    chips = _lag_post_close_source_frame(chips_raw, market.index)
    margin = _lag_post_close_source_frame(margin_raw, market.index)
    futures = _lag_post_close_source_frame(futures_raw, market.index)

    def build_features(chips_frame, margin_frame, futures_frame):
        frames = {}
        for horizon in (1, 2, 3):
            X, _, forward_returns = pipeline.build(
                market,
                chips=chips_frame,
                margin=margin_frame,
                futures=futures_frame,
                h=horizon,
            )
            frames[horizon] = (X, forward_returns)
        return frames

    # Determine the earliest date at which the complete feature set is valid,
    # then use that same calendar and expanding training history for every
    # ablation group. This prevents the futures layer's shorter history from
    # being mistaken for a model improvement or deterioration.
    full_features_raw = build_features(chips, margin, futures)
    full_d1 = full_features_raw[1][0].apply(pd.to_numeric, errors="coerce")
    full_valid_dates = full_d1.index[full_d1.notna().all(axis=1)]
    if not len(full_valid_dates):
        raise RuntimeError("完整特徵組沒有可用的共同起始交易日")
    common_start = full_valid_dates[0]

    def trim_features(frames, group_name):
        trimmed = {}
        for horizon, (X, forward_returns) in frames.items():
            X = X.loc[X.index >= common_start].copy()
            missing_rows = int(X.isna().any(axis=1).sum())
            if missing_rows:
                raise RuntimeError(
                    f"{group_name} D+{horizon} 在共同起始日 {common_start.date()} 後仍有 "
                    f"{missing_rows} 個特徵缺值交易日；不能做不公平的消融比較"
                )
            trimmed[horizon] = (X, forward_returns.reindex(X.index))
        return trimmed

    feature_groups = {
        "price_only": trim_features(
            build_features(None, None, None), "price_only"
        ),
        "chips_margin": trim_features(
            build_features(chips, margin, None), "chips_margin"
        ),
        "chips_margin_futures": trim_features(
            full_features_raw, "chips_margin_futures"
        ),
    }

    group_results = {}
    for group_name, group_features in feature_groups.items():
        group_results[group_name] = build_regime_router_forecasts(
            group_features,
            close,
            target_meta={},
            market_data=market,
            post_close_features_lagged=True,
        )

    # Assert identical OOS windows for every feature group and horizon before
    # calculating incremental feature value.
    alignment = {}
    for horizon in (1, 2, 3):
        signatures = {}
        for group_name, group_result in group_results.items():
            item = next(
                (session for session in group_result.get("sessions", [])
                 if session.get("horizon_sessions") == horizon),
                {},
            )
            metrics = item.get("routed_forecast_validation", {})
            signatures[group_name] = {
                "oos_samples": metrics.get("oos_samples", 0),
                "start_date": metrics.get("oos_start_date"),
                "end_date": metrics.get("oos_end_date"),
            }
        unique = {
            (value["oos_samples"], value["start_date"], value["end_date"])
            for value in signatures.values()
        }
        if len(unique) != 1:
            raise RuntimeError(
                f"D+{horizon} feature ablation 的 OOS 日期未對齊："
                + json.dumps(signatures, ensure_ascii=False)
            )
        alignment[f"D+{horizon}"] = next(iter(signatures.values()))

    result = group_results["chips_margin_futures"]
    features_by_horizon = feature_groups["chips_margin_futures"]

    def summarize_group(group_result):
        summary = {}
        for session in group_result.get("sessions", []):
            validation = session.get("router_validation", {})
            forecast = session.get("routed_forecast_validation", {})
            label = session.get("label", f"D+{session.get('horizon_sessions')}")
            summary[label] = {
                "oos_samples": forecast.get("oos_samples"),
                "oos_start_date": forecast.get("oos_start_date"),
                "oos_end_date": forecast.get("oos_end_date"),
                "macro_f1": validation.get("macro_f1"),
                "high_extreme_recall": validation.get("high_extreme_recall"),
                "high_extreme_precision": validation.get("high_extreme_precision"),
                "routed_mae_points": forecast.get("routed_mae_points"),
                "global_ridge_mae_points": forecast.get("global_ridge_mae_points"),
                "routed_minus_global_mae_points": (
                    round(forecast["routed_mae_points"] - forecast["global_ridge_mae_points"], 2)
                    if forecast.get("routed_mae_points") is not None
                    and forecast.get("global_ridge_mae_points") is not None else None
                ),
                "max_daily_move_mae_pct": forecast.get("max_daily_move_mae_pct"),
                "global_max_daily_move_mae_pct": forecast.get("global_max_daily_move_mae_pct"),
            }
        return summary

    ablation_groups = {
        name: summarize_group(group_result)
        for name, group_result in group_results.items()
    }
    price_reference = ablation_groups["price_only"]
    for group_name in ("chips_margin", "chips_margin_futures"):
        for label, metrics in ablation_groups[group_name].items():
            base = price_reference.get(label, {})
            metrics["global_mae_delta_vs_price_only"] = (
                round(metrics["global_ridge_mae_points"] - base["global_ridge_mae_points"], 2)
                if metrics.get("global_ridge_mae_points") is not None
                and base.get("global_ridge_mae_points") is not None else None
            )
            metrics["macro_f1_delta_vs_price_only"] = (
                round(metrics["macro_f1"] - base["macro_f1"], 4)
                if metrics.get("macro_f1") is not None and base.get("macro_f1") is not None else None
            )
            metrics["high_extreme_recall_delta_vs_price_only"] = (
                round(metrics["high_extreme_recall"] - base["high_extreme_recall"], 4)
                if metrics.get("high_extreme_recall") is not None
                and base.get("high_extreme_recall") is not None else None
            )

    result["feature_ablation"] = {
        "common_feature_start_date": str(common_start.date()),
        "oos_window_alignment": alignment,
        "groups": ablation_groups,
        "note": (
            "價格層、法人＋融資融券、法人＋融資融券＋期貨三組使用相同起始特徵日期、"
            "相同台灣交易日曆與完全相同的 OOS 日期；增量差異才可歸因於特徵組合。"
        ),
    }

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
