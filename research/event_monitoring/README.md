# Event-driven monitoring research v1

Status: research-only design. This directory does not change production forecasts, signals.json, the live dashboard, or scheduled refresh jobs.

## Existing capabilities and gaps

The repository already contains news.py (Google News/Bing RSS discovery, CNA and government RSS feeds, a small institutional-analyst watchlist, policy-event extraction and daily snapshots), policy_events.jsonl (URL-derived event IDs and first/last-seen timestamps), news_research_history.jsonl (daily article snapshots and derived features), and news_history_import.py (a separate GDELT historical-news experiment using an 08:30 Asia/Taipei cutoff).

The main gap for option C is that discovery is currently topic/search-query-led. There is no maintained named-person/entity watchlist or normalized event record shared across sources. Article publication times are mostly retained as raw source strings; first/last-seen timestamps are currently applied to extracted policy events, not consistently to every article. Existing news scores are not validated as incremental predictors and must remain outside production forecasts.

## Design principles

1. Point-in-time first: preserve source-provided publication time as raw text and normalized UTC only when unambiguous. Separately record first observation and every later fetch. Never replace publication time with crawl time.
2. Evidence before interpretation: separate source item and factual summary from AI interpretation. Distinguish official confirmation, attributed reporting, allegation, unverified claim, correction and retraction.
3. Entity plus global discovery: use both a curated watchlist and broad topic discovery. A watchlist hit is not automatically market-relevant; a major event can matter without mentioning a watched person.
4. No sentiment shortcut: classify event, affected channels/sectors, transmission mechanism, horizon and uncertainty. Direction may be mixed or unknown. Do not convert headline tone directly into a TAIEX return forecast.
5. Deduplicate and version: cluster syndicated copies and updates but preserve every source item and correction. Material updates create a new version rather than overwriting prior records.
6. No rumor laundering: AI summaries retain attribution and uncertainty. Allegations and speculation must not be presented as established fact.
7. Rights and cost: prefer official/open feeds and metadata. Respect source terms, rate limits, attribution and retention rules. Do not scrape social platforms outside published interfaces.

## Proposed source registry

| Source | Intended use | Initial treatment |
|---|---|---|
| TWSE OpenAPI | Taiwan market context, exchange announcements/news and reference data | First-party source; check endpoint-specific terms and update timing |
| Taiwan government open data | Exchange news dataset and official market information | Open Government Data License v1.0 is listed for the dataset; retain attribution and license metadata |
| FSC, Executive Yuan, MOEA, CBC, USTR, Federal Reserve, White House | Policy decisions, official statements, speeches and releases | First-party evidence; capture exact release URL and publication timestamp |
| CNA RSS | Taiwan political, finance, technology and international news discovery | Secondary reporting; verify consequential claims against primary sources |
| Yahoo Finance RSS | Taiwan/global market and company-news discovery | Secondary reporting; use only within feed terms and retain links/attribution |
| Company investor relations (e.g. TSMC, NVIDIA, AMD) | Earnings, guidance, capex, orders, executive changes and official statements | First-party corporate source |
| X public posts | Statements by monitored public figures | Optional only through X published API/interfaces and applicable developer terms; no scraping. Access is usage-billed, so keep disabled until budget and retention policy are approved |
| GDELT DOC 2.0 | Broad global news discovery and limited-window historical metadata research | Discovery/backfill aid, not ground truth; preserve its seen-time semantics and query-cap flags |

### Source-selection rule

For a material claim, prefer (1) official primary document or original public statement, (2) multiple independent reputable reports, (3) a single attributed report, then (4) unverified social content. Lower evidence levels may trigger monitoring but must not be described as confirmed. Search aggregators are discovery tools, not sources of truth.

## Initial event taxonomy

- policy_trade: tariffs, export controls, sanctions, industrial policy
- monetary_macro: central-bank decisions, rates, inflation, FX, fiscal policy
- geopolitics: diplomatic/security developments, conflict, cross-border restrictions
- semiconductor_ai: chips, AI infrastructure, foundry capacity, packaging, equipment, supply chain
- corporate: earnings, guidance, capex, orders, litigation, investigation, management changes
- market_structure: exchange rules, index changes, liquidity or market-wide disruptions
- shock: natural disaster, major outage, logistics or energy disruption
- other: retained for review; not automatically used as a model feature

## Minimum event lifecycle

1. Collect source item with raw timestamp, URL, source identity, language and fetch timestamp.
2. Normalize timestamps to UTC only when source timezone is known; otherwise normalized time is null and the reason is recorded.
3. Deduplicate exact URLs and near-identical copies; assign a cluster ID without deleting source items.
4. Extract entities and taxonomy; keep extraction model/version and confidence.
5. Assess evidence state, Taiwan relevance, sector/channel mapping, expected horizon and volatility potential. Allow unknown and mixed.
6. Human-review high-impact allegations, conflicting reports and low-confidence events before they can become model features.
7. Freeze a point-in-time snapshot at forecast cutoff. Only items first observed at or before that cutoff are eligible.
8. Label outcomes only after the relevant TAIEX session closes. Never backfill later corrections into earlier snapshots.

## Watchlist policy

The starter watchlist is deliberately small: Donald Trump and Jensen Huang (the user's examples); TSMC leadership as a role-based entity; and the Federal Reserve Chair as a role-based entity. Resolve office-holders using dated official records rather than hard-coding timeless identities. Add people only where a plausible Taiwan-market transmission channel exists. Aliases and roles need effective dates; ambiguous name matches remain unresolved. The watchlist is a discovery aid, not a ranking of personal importance.

## Backtest gate before model integration

Keep the existing Ridge forecast as the baseline. Compare (a) market-only baseline, (b) baseline plus simple timestamp-safe event counts/categories, and (c) baseline plus event features with source/evidence/novelty controls. Use the same dates and 08:30 Asia/Taipei information cutoff, with expanding/walk-forward evaluation and horizon-appropriate purging. Report D+1/D+2/D+3 separately: MAE, directional calibration where relevant, event-day vs non-event-day performance, volatility calibration, missingness and source coverage. Include ablations by event type and source class. Do not accept a model solely because of in-sample fit or one short holdout.

## Implementation boundary

This PR adds design/configuration artifacts only. It does not add collectors, call external APIs, introduce credentials, modify GitHub Actions, write to Supabase, change Render, or alter production forecast outputs.