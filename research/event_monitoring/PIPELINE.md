# Event monitoring pipeline v1 (research only)

This module is a first executable step after the design/schema PR. It deliberately starts with deterministic extraction and no AI inference so that timestamp, deduplication, provenance and evidence rules can be tested independently.

## Run locally

\`\`\`bash
python event_monitor.py
python -m unittest discover -s tests -p 'test_event_monitor.py' -v
\`\`\`

The script reuses \`news.headlines()\`, so it uses the RSS/search discovery already implemented in \`news.py\`. It writes only to the local \`research_data/event_monitor_history.jsonl\` file by default. No GitHub Actions workflow invokes it.

## Current behavior

- Reuses the existing Google News/Bing RSS, CNA and government feed discovery.
- Applies the starter person/role watchlist.
- Parses timezone-aware RSS publication timestamps; preserves raw timestamps and refuses to guess an unprovided timezone.
- Records first-seen, last-seen and forecast-eligibility timestamps separately.
- Assigns conservative rule-based event categories and evidence states.
- Flags allegations and investigation-related items for follow-up.
- Keeps direction and volatility as unknown; it does not generate a market prediction.
- Rejects missing/non-HTTP source URLs and keeps source attribution.
- Stores a local JSONL research archive only.

## Known limitations

- No AI classification, source corroboration, semantic clustering, source licensing automation or human-review interface yet.
- RSS publication time can be missing, malformed, updated by a publisher or not equivalent to first publication. First-seen time is the system's own observation time.
- Exact-title clustering is only a first-pass duplicate hint, not a claim that two articles describe the same event.
- A domain ending in a government suffix is only a preliminary source-class heuristic; it does not validate the truth of every statement on the page.
- Rule-based entity matching can miss aliases and can produce false positives. Role names require later dated official resolution.
- The event records are research candidates, not approved model features.

## Validation gate

Before connecting this to a forecast:
1. Run unit tests and schema validation.
2. Inspect a sample of live records manually for entity, source, evidence and timestamp errors.
3. Add AI extraction only behind a separately versioned adapter; retain raw records and model/prompt versions.
4. Measure source coverage, false positives, missing timestamps, duplicate precision and review workload.
5. Build point-in-time snapshots before evaluating any incremental predictive value.
6. Compare against the market-only Ridge baseline on identical forecast cutoffs and walk-forward dates.

No production data, Supabase, Render, website, credentials or scheduled workflows are changed by this PR.
