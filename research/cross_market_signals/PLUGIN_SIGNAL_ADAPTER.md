# Cross-market and provider-derived signal adapter (research only)

## Purpose

Evaluate external market-context and media-sentiment signals as additions to the TAIEX market-only Ridge baseline. This is a research specification, not a production integration or forecast. Do not use provider sentiment or narrative text as a direct TAIEX price target.

## Candidate signal families

### 1. Cross-market price and risk context

Candidate daily features, calculated from point-in-time observations:

- US broad market: S&P 500 / SPY, NASDAQ Composite / NASDAQ 100 / QQQ.
- Semiconductor complex: SOX or a semiconductor ETF proxy, plus NVDA, AMD, AVGO and TSM ADR returns.
- Regional confirmation: Korea and Japan equity proxies; Taiwan ETF only as a market proxy with its own close timestamp.
- Risk regime: VIX level/change, US Treasury 2Y/10Y yield changes, USD/TWD and USD/KRW changes.
- Optional macro context: oil and copper returns, kept separate from equity momentum.

Use returns and changes, not raw price levels. Prefer 1D, 5D and volatility measures. Record instrument, currency, venue, session close, provider, source timestamp, retrieval timestamp and adjusted/unadjusted status. ETF returns are proxies, not the underlying index return.

### 2. Provider-derived media sentiment

A Bigdata/RavenPack company sentiment snapshot can expose:

- current sentiment score, baseline, momentum and standardized z-scores;
- media-attention momentum and attention z-scores;
- document count and article-level source/timestamp evidence.

Initial watchlist candidates: NVIDIA, AMD, Broadcom and TSMC. Keep company-level sentiment separate; do not average it into a single opaque "market sentiment" score. Derive transparent, pre-specified features such as lagged sentiment level/change, attention change, and dispersion across covered companies only after snapshots have been archived.

The plugin's sentiment score is an NLP/media-tone measure, not a calibrated probability of a stock or TAIEX rise/fall. Its generated narrative is supporting context for review, not a model feature. Allegations, investigations, opinion, duplicate syndicated coverage and corrections must be tagged and reviewed; negative wording alone is not evidence that an allegation is true.

### 3. Other provider signals

Alpha Vantage news sentiment and technical indicators may be explored as alternative sources. Current test access to the NEWS_SENTIMENT endpoint returned a premium/rate-limit response; no usable observation was obtained. Do not silently replace missing values with zeros or substitute another provider without recording the source change.

## Point-in-time record contract

Every provider snapshot must preserve:

- provider and endpoint/tool name;
- provider/model version where available;
- requested entities/instruments and parameters;
- provider's `as_of` or article publication timestamps;
- `retrieved_at_utc`;
- `forecast_eligible_at_utc` (no earlier than retrieval time and any relevant publication/market-close time);
- raw response hash and immutable raw payload location;
- normalized values, units, currency, missingness and quality flags;
- source article IDs/URLs and correction/version lineage where applicable.

Never backfill today's provider score into prior historical dates. A current 90-day aggregate is not a 90-day historical time series. Historical evaluation requires archived snapshots that were actually available at each forecast cutoff, or a separately sourced historical dataset with reliable point-in-time timestamps.

## Plugin-to-website boundary

ChatGPT plugin results available in a conversation are not automatically callable from the Render-hosted website. Do not put ChatGPT-session output, OAuth assumptions, or user credentials into the deployed app. A production-capable adapter needs a separately authorized server-side provider API, documented terms, server-side secret storage, rate-limit handling and health monitoring. Until that exists, plugin results can be used for research design and manual sample review only.

## Evaluation plan

1. Capture timestamped snapshots without affecting production forecasts.
2. Measure coverage, missingness, update latency, revisions, duplicate/syndication concentration and entity-match quality.
3. Construct lagged candidate features with strict forecast-cutoff joins.
4. Compare market-only Ridge against market + cross-market, market + sentiment, and combined ablations on identical expanding-window walk-forward dates.
5. Report D+1/D+2/D+3 MAE against flat baseline, directional metrics, calibration, regime/extreme-event behavior, and incremental performance with uncertainty.
6. Reject features that do not improve out-of-sample performance consistently or whose historical availability cannot be proven.

No provider-derived signal in this document is approved for production use.
