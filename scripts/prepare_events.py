"""
Build event-level dataset from news_clean.csv and sp500_5y_prices.csv.

Steps:
1) load data
2) parse publication date (day resolution)
3) whitelist-based ticker matching
4) explode news-to-ticker to one row per (doc, ticker)
5) align with next trading-day return as label
6) save news_events.parquet / news_events.csv

All code in English; explanations will be provided separately.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

NEWS_PATH = Path("data/news_clean.csv")
PRICE_PATH = Path("data/sp500_5y_prices.csv")
OUT_PARQUET = Path("data/news_events.parquet")
OUT_CSV = Path("data/news_events.csv")

# Strict whitelist to reduce false positives
WHITELIST = {
    "RF": r"\bRegions Financial\b|\bRegions Bank\b|\bRF\b",
    "CFG": r"\bCitizens Financial\b|\bCitizens Bank\b|\bCFG\b",
    "TFC": r"\bTruist\b|\bTruist Financial\b|\bTFC\b",
    "FITB": r"\bFifth Third\b|\bFifth Third Bancorp\b|\bFITB\b",
    "HBAN": r"\bHuntington Bancshares\b|\bHuntington Bank\b|\bHBAN\b",
    "ITW": r"\bIllinois Tool Works\b|\bITW\b",
    "DOV": r"\bDover Corp\b|\bDover Corporation\b|\bDOV\b",
    "WAB": r"\bWabtec\b|\bWestinghouse Air Brake\b|\bWAB\b",
    "IR": r"\bIngersoll Rand\b|\bIR\b",
}

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December"
)
DATE_RE = re.compile(rf"(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})", re.IGNORECASE)


def parse_date(meta: str) -> Optional[pd.Timestamp]:
    if not isinstance(meta, str):
        return None
    m = DATE_RE.search(meta)
    if not m:
        return None
    day, mon, year = m.group(1), m.group(2), m.group(3)
    try:
        return pd.to_datetime(f"{day} {mon} {year}", format="%d %B %Y")
    except Exception:
        return None


def find_tickers(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    hits = []
    for tk, pat in WHITELIST.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(tk)
    return hits


def load_prices() -> pd.DataFrame:
    prices = pd.read_csv(PRICE_PATH)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.set_index("Date").sort_index()
    return prices


def compute_next_return(prices: pd.DataFrame) -> pd.DataFrame:
    ret = prices.pct_change()  # simple return
    next_ret = ret.shift(-1)   # t+1 return
    return next_ret


def next_trading_return(next_ret: pd.DataFrame, dt: pd.Timestamp, tk: str) -> Optional[float]:
    try:
        series = next_ret.loc[next_ret.index > dt, tk]
        if series.empty:
            return None
        return series.iloc[0]
    except Exception:
        return None


def main():
    news = pd.read_csv(NEWS_PATH)
    news["date"] = news["meta"].apply(parse_date)
    news["text"] = news["title"].fillna("") + ". " + news["body"].fillna("")
    news["tickers"] = news["text"].apply(find_tickers)

    before_filter = len(news)
    news = news[news["tickers"].map(len) > 0]
    after_filter = len(news)

    # explode to one row per ticker
    news = news.explode("tickers").rename(columns={"tickers": "ticker"})

    prices = load_prices()
    next_ret = compute_next_return(prices)

    news["return_next"] = [
        next_trading_return(next_ret, dt, tk) for dt, tk in zip(news["date"], news["ticker"])
    ]
    news = news.dropna(subset=["return_next"])
    news["label"] = (news["return_next"] > 0).astype(int)

    # select compact columns
    out_cols = ["doc_id", "filename", "date", "ticker", "title", "meta", "text", "return_next", "label"]
    news_out = news[out_cols]

    news_out.to_parquet(OUT_PARQUET, index=False)
    news_out.to_csv(OUT_CSV, index=False)

    # summary print
    per_ticker = news_out["ticker"].value_counts().to_dict()
    print(f"Input articles: {before_filter}, after ticker filter: {after_filter}, exploded rows: {len(news_out)}")
    print("Rows per ticker:", per_ticker)
    print(f"Saved: {OUT_PARQUET} and {OUT_CSV}")


if __name__ == "__main__":
    main()
