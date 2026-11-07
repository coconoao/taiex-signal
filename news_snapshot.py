"""Collect RSS headlines and save neutral research snapshots only.

This deliberately does not run the optional model-analysis or forecasting
providers in news.py; it only captures source metadata and neutral features.
"""
import news


def main():
    headlines = news.headlines()
    if len(headlines) < 3:
        raise SystemExit(
            f"Only {len(headlines)} headlines collected; keeping the previous snapshot."
        )
    news.save_research_snapshot(headlines)
    print(
        "News snapshot collected:",
        len(headlines),
        "headlines from",
        len({str(x.get("source") or x.get("tag") or "").strip() for x in headlines
             if str(x.get("source") or x.get("tag") or "").strip()}),
        "sources",
        flush=True,
    )


if __name__ == "__main__":
    main()
