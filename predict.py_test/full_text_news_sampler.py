import os
import re
import sys
import json
import time
import random
import sqlite3
import subprocess
from datetime import datetime, timezone, timedelta

import feedparser
import requests
import pandas as pd
from dateutil import parser as dateparser

# Full-text extraction (fast + robust)
import trafilatura


# -----------------------
# Config
# -----------------------
DAYS = 30
REAL_TARGET = 160
MISINFO_TARGET = 40

MIN_WORDS_REAL = 400
MIN_WORDS_MISINFO = 180

REQUEST_TIMEOUT = 20
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
HEADERS = {"User-Agent": USER_AGENT}

# Speed controls for MacBook Air
MAX_CANDIDATES_PER_SOURCE = 700   # limit index size per feed
SLEEP_BETWEEN_REQUESTS = 0.1
SLEEP_BETWEEN_PREDICTS = 0.01

random.seed(42)


# -----------------------
# Sources
# type = "rss" uses feed directly
# type = "gnews" uses Google News RSS "search" query to index a domain
# -----------------------
REAL_SOURCES = [
    {"name": "BBC", "type": "rss", "feed": "https://feeds.bbci.co.uk/news/rss.xml"},
    {"name": "NPR", "type": "rss", "feed": "https://feeds.npr.org/1001/rss.xml"},
    {"name": "Guardian_World", "type": "rss", "feed": "https://www.theguardian.com/world/rss"},

    # PBS NewsHour RSS feeds (headlines feed is solid for volume)
    {"name": "PBS_Headlines", "type": "rss", "feed": "https://www.pbs.org/newshour/feeds/rss/headlines"},
]

MISINFO_SOURCES = [
    # NaturalNews: use your working RSS
    {"name": "NaturalNews", "type": "rss", "feed": "https://www.naturalnews.com/feedlyrss.xml"},

    # These don't have reliable RSS you can depend on—index via Google News RSS domain search
    {"name": "InfoWars", "type": "gnews", "domain": "infowars.com"},
    {"name": "GatewayPundit", "type": "gnews", "domain": "thegatewaypundit.com"},
]

# Google News RSS index pattern (domain-scoped, last 30d)
# Example the community uses: https://news.google.com/rss/search?q=when:30d+allinurl:apnews.com :contentReference[oaicite:8]{index=8}
def gnews_rss_for_domain(domain: str, days: int = 30) -> str:
    return f"https://news.google.com/rss/search?q=when:{days}d+allinurl:{domain}&hl=en-US&gl=US&ceid=US:en"


# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # predict.py_test
OUT_DIR = os.path.join(BASE_DIR, "eval_outputs_fulltext")
os.makedirs(OUT_DIR, exist_ok=True)

DB_PATH = os.path.join(OUT_DIR, "news_eval.db")
CSV_PATH = os.path.join(OUT_DIR, "predictions.csv")

PREDICT_SCRIPT = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "predict.py"))


# -----------------------
# SQLite
# -----------------------
def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actual_bucket TEXT NOT NULL,   -- real | misinfo
            source TEXT NOT NULL,
            url TEXT UNIQUE,
            title TEXT,
            published TEXT,
            full_text TEXT,
            word_count INTEGER,
            scraped_at TEXT DEFAULT (datetime('now'))
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            predicted_label TEXT,
            full_text_fake_prob REAL,
            aggregated_fake_prob REAL,
            explanation TEXT,
            top_snippets TEXT,
            raw_output TEXT,
            ran_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(article_id) REFERENCES articles(id)
        );
    """)
    conn.commit()


def upsert_article(conn, bucket, source, url, title, published, full_text, word_count) -> int:
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO articles (actual_bucket, source, url, title, published, full_text, word_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (bucket, source, url, title, published, full_text, word_count))
    conn.commit()
    cur.execute("SELECT id FROM articles WHERE url = ?", (url,))
    return int(cur.fetchone()[0])


# -----------------------
# Helpers
# -----------------------
def parse_feed(feed_url: str):
    r = requests.get(feed_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return feedparser.parse(r.text)

def published_within_days(published_str: str, days: int) -> bool:
    if not published_str:
        return True
    try:
        dt = dateparser.parse(published_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return dt >= cutoff
    except Exception:
        return True

def canonicalize_url(url: str) -> str:
    return url.split("#")[0].strip()

def scrape_full_text(url: str) -> str | None:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if not text:
            return None
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return None

def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text or ""))


# -----------------------
# Candidate collection
# -----------------------
def collect_candidates_for_source(source: dict) -> list[dict]:
    if source["type"] == "rss":
        feed_url = source["feed"]
    elif source["type"] == "gnews":
        feed_url = gnews_rss_for_domain(source["domain"], DAYS)
    else:
        raise ValueError("Unknown source type")

    parsed = parse_feed(feed_url)
    items = []

    for e in (parsed.entries or [])[:MAX_CANDIDATES_PER_SOURCE]:
        url = getattr(e, "link", None)
        if not url:
            continue
        url = canonicalize_url(url)

        title = getattr(e, "title", "") or ""
        published = getattr(e, "published", None) or getattr(e, "updated", None) or ""

        if not published_within_days(published, DAYS):
            continue

        items.append({
            "source": source["name"],
            "url": url,
            "title": title,
            "published": published
        })

    return items


# -----------------------
# Stratified random sampling
# -----------------------
def stratified_sample(candidates_by_source: dict[str, list[dict]], target_n: int) -> list[dict]:
    # First pass: equal quota per source (rounded down), then fill remainder from pooled leftovers
    sources = list(candidates_by_source.keys())
    if not sources:
        return []

    per_source = max(1, target_n // len(sources))
    picked = []
    leftovers = []

    for s in sources:
        cand = candidates_by_source[s][:]
        random.shuffle(cand)
        take = cand[:per_source]
        rest = cand[per_source:]
        picked.extend(take)
        leftovers.extend(rest)

    if len(picked) > target_n:
        random.shuffle(picked)
        return picked[:target_n]

    remaining = target_n - len(picked)
    random.shuffle(leftovers)
    picked.extend(leftovers[:remaining])
    return picked


# -----------------------
# predict.py parsing (minimal; refine later)
# -----------------------
P_LABEL = re.compile(r"Label:\s*(FAKE|REAL)\b", re.IGNORECASE)
P_FULL = re.compile(r"Full-text fake probability:\s*([0-9]*\.?[0-9]+)\s*%", re.IGNORECASE)

def run_predict_py(text: str) -> tuple[dict, str]:
    proc = subprocess.run(
        [sys.executable, PREDICT_SCRIPT],
        input=text,
        text=True,
        capture_output=True
    )
    raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    label = None
    m = P_LABEL.search(raw)
    if m:
        label = m.group(1).upper()

    full_prob = None
    m = P_FULL.search(raw)
    if m:
        full_prob = float(m.group(1)) / 100.0

    return {"predicted_label": label, "full_text_fake_prob": full_prob}, raw


def save_prediction(conn, article_id: int, parsed: dict, raw: str) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (article_id, predicted_label, full_text_fake_prob, raw_output, top_snippets)
        VALUES (?, ?, ?, ?, ?)
    """, (article_id, parsed.get("predicted_label"), parsed.get("full_text_fake_prob"), raw, json.dumps([])))
    conn.commit()


# -----------------------
# Main pipeline
# -----------------------
def ingest_bucket(conn, bucket_name: str, sources: list[dict], target_n: int, min_words: int) -> int:
    MAX_PASSES = 6                   # try multiple times to reach target
    POOL_MULTIPLIER = 12             # build a large candidate pool vs target
    inserted_urls = set()

    # Pull existing URLs already in DB for this bucket so reruns don't duplicate
    existing = pd.read_sql_query(
        "SELECT url FROM articles WHERE actual_bucket = ?",
        conn,
        params=(bucket_name,)
    )
    for u in existing["url"].tolist():
        inserted_urls.add(u)

    inserted = len(inserted_urls)

    for _pass in range(MAX_PASSES):
        if inserted >= target_n:
            break

        # Collect candidates from each source
        candidates = []
        for src in sources:
            cands = collect_candidates_for_source(src)
            candidates.extend(cands)

        # Dedupe pool
        uniq = {}
        for c in candidates:
            uniq[c["url"]] = c
        pool = list(uniq.values())

        # Shuffle for random sampling
        random.shuffle(pool)

        # Limit pool size so you don't scrape forever
        max_pool = min(len(pool), target_n * POOL_MULTIPLIER)
        pool = pool[:max_pool]

        for item in pool:
            if inserted >= target_n:
                break

            url = item["url"]
            if url in inserted_urls:
                continue

            text = scrape_full_text(url)
            time.sleep(SLEEP_BETWEEN_REQUESTS)

            if not text:
                continue

            wc = word_count(text)

            # Quality filters
            if wc < min_words:
                continue

            upsert_article(
                conn,
                bucket_name,
                item["source"],
                url,
                item.get("title", ""),
                item.get("published", ""),
                text,
                wc
            )
            inserted_urls.add(url)
            inserted += 1

        # If we made zero progress in a pass, no point continuing
        # (usually means sources blocked or too few candidates)
        if inserted < target_n and _pass >= 1:
            # If progress is extremely slow, you can keep going,
            # but this prevents infinite looping when sources are dead.
            pass

    return inserted


def compute_metrics(df: pd.DataFrame) -> tuple[float, float]:
    # False positive rate on REAL (predicted FAKE when actual is real)
    real = df[df["actual_bucket"] == "real"]
    fpr = float("nan") if len(real) == 0 else (real["predicted_label"] == "FAKE").mean()

    # Overall accuracy with mapping: real->REAL, misinfo->FAKE (weakly meaningful if misinfo bucket is truly misinfo)
    correct = (
        ((df["actual_bucket"] == "real") & (df["predicted_label"] == "REAL")) |
        ((df["actual_bucket"] == "misinfo") & (df["predicted_label"] == "FAKE"))
    )
    acc = float("nan") if len(df) == 0 else correct.mean()
    return fpr, acc


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    got_real = ingest_bucket(conn, "real", REAL_SOURCES, REAL_TARGET, MIN_WORDS_REAL)
    got_mis  = ingest_bucket(conn, "misinfo", MISINFO_SOURCES, MISINFO_TARGET, MIN_WORDS_MISINFO)

    print(f"Collected full-text real: {got_real}/{REAL_TARGET}, misinfo: {got_mis}/{MISINFO_TARGET}")

    df_articles = pd.read_sql_query("""
        SELECT id, actual_bucket, source, url, title, published, full_text, word_count
        FROM articles
        ORDER BY id ASC
    """, conn)

    print(f"Running predict.py on {len(df_articles)} rows...")

    for _, row in df_articles.iterrows():
        parsed, raw = run_predict_py(row["full_text"])
        save_prediction(conn, int(row["id"]), parsed, raw)
        time.sleep(SLEEP_BETWEEN_PREDICTS)

    df = pd.read_sql_query("""
        SELECT
            p.id AS prediction_id,
            a.actual_bucket,
            p.predicted_label,
            a.source,
            a.url,
            a.title,
            a.published,
            a.word_count,
            p.full_text_fake_prob,
            p.ran_at
        FROM predictions p
        JOIN articles a ON a.id = p.article_id
        ORDER BY p.id DESC
    """, conn)

    df.to_csv(CSV_PATH, index=False)

    fpr, acc = compute_metrics(df)
    print(f"Saved DB file: {DB_PATH}")
    print(f"Saved predictions.csv: {CSV_PATH}")
    print(f"False Positive Rate on REAL: {fpr:.3f}")
    print(f"Overall accuracy (real->REAL, misinfo->FAKE): {acc:.3f}")
    print(df.head(10).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
