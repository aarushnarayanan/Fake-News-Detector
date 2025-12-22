import os 
import re 
import sys
import json
import time
import sqlite3
import subprocess
import feedparser
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

# =========================
# Your RSS links
# =========================

REAL_FEEDS = [
    ("BBC", "https://feeds.bbci.co.uk/news/rss.xml"),
    ("NYT_World", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ('AP', "https://news.google.com/rss/search?q=when:30d+allinurl:apnews.com&hl=en-US&gl=US&ceid=US:en")
]

MISINFO_FEEDS = [
    ("PolitiFact_Factchecks", "https://www.politifact.com/rss/factchecks/"),
]
# =========================
# Targets
# =========================
TOTAL_REAL = 90
TOTAL_MISINFO = 10
MAX_DAYS_OLD = 30

# Speed controls (Macbook-friendly)
MAX_CHARS_REAL = 4000
MAX_CHARS_MISINFO = 2500
SLEEP_BETWEEN_PREDICTS = 0.01
REQUEST_TIMEOUT = 20

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
HEADERS = {"User-Agent": USER_AGENT}


# =========================
# Paths: put DB + CSV under predict.py_test/eval_outputs/
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder containing this script (predict.py_test)
OUT_DIR = os.path.join(BASE_DIR, "eval_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DB_PATH = os.path.join(OUT_DIR, "news_eval.db")
CSV_PATH = os.path.join(OUT_DIR, "predictions.csv")

# predict.py is in ../models/predict.py (relative to predict.py_test)
PREDICT_SCRIPT = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "predict.py"))


# =========================
# SQLite setup
# =========================
def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bucket TEXT NOT NULL,          -- 'real' or 'misinfo'
            source TEXT NOT NULL,          -- BBC, NYT_World, PolitiFact_Factchecks
            url TEXT UNIQUE,
            title TEXT,
            published TEXT,
            content TEXT NOT NULL,
            scraped_at TEXT DEFAULT (datetime('now'))
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            label TEXT,
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


def upsert_article(conn, bucket, source, url, title, published, content) -> int:
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO articles (bucket, source, url, title, published, content)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (bucket, source, url, title, published, content))
    conn.commit()

    cur.execute("SELECT id FROM articles WHERE url = ?", (url,))
    row = cur.fetchone()
    return int(row[0])


# =========================
# RSS parsing via requests (avoids 0 entries)
# =========================
def parse_feed_with_requests(feed_url: str):
    r = requests.get(feed_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return feedparser.parse(r.text)


def within_days(published_str: str, max_days: int) -> bool:
    if not published_str:
        return True
    try:
        dt = dateparser.parse(published_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
        return dt >= cutoff
    except Exception:
        return True


def clean_text(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)     # strip HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s


def collect_from_feeds(conn, bucket: str, feeds: list[tuple[str, str]], target_n: int, max_chars: int) -> int:
    got = 0
    seen_urls = set()

    for source_name, feed_url in feeds:
        if got >= target_n:
            break

        parsed = parse_feed_with_requests(feed_url)
        entries = parsed.entries or []

        for e in entries:
            if got >= target_n:
                break

            url = getattr(e, "link", None)
            if not url or url in seen_urls:
                continue

            published = getattr(e, "published", None) or getattr(e, "updated", None) or ""
            if not within_days(published, MAX_DAYS_OLD):
                continue

            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            summary = clean_text(summary)

            # RSS-based "text blob"
            text_blob = f"{title}\n\n{summary}".strip()
            text_blob = text_blob[:max_chars]

            # Basic quality filters
            if bucket == "real" and len(text_blob) < 200:
                continue
            if bucket == "misinfo" and len(text_blob) < 120:
                continue

            upsert_article(conn, bucket, source_name, url, title, published, text_blob)
            seen_urls.add(url)
            got += 1

    return got


# =========================
# Running models/predict.py unchanged
# =========================
P_LABEL = re.compile(r"Label:\s*(FAKE|REAL)\b", re.IGNORECASE)
P_FULL = re.compile(r"Full-text fake probability:\s*([0-9]*\.?[0-9]+)\s*%", re.IGNORECASE)
P_AGG  = re.compile(r"Aggregated fake probability:\s*([0-9]*\.?[0-9]+)\s*%", re.IGNORECASE)

def run_predict_py(text: str) -> tuple[dict, str]:
    proc = subprocess.run(
        [sys.executable, PREDICT_SCRIPT],
        input=text,
        text=True,
        capture_output=True
    )
    raw = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    parsed = parse_predict_output(raw)
    return parsed, raw


def parse_predict_output(raw: str) -> dict:
    # (We’ll improve parsing after you share raw_output)
    label = None
    m = P_LABEL.search(raw)
    if m:
        label = m.group(1).upper()

    full_prob = None
    m = P_FULL.search(raw)
    if m:
        full_prob = float(m.group(1)) / 100.0

    agg_prob = None
    m = P_AGG.search(raw)
    if m:
        agg_prob = float(m.group(1)) / 100.0

    return {
        "label": label,
        "full_text_fake_prob": full_prob,
        "aggregated_fake_prob": agg_prob,
        "explanation": None,
        "top_snippets": [],
    }


def save_prediction(conn, article_id: int, parsed: dict, raw: str) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (
            article_id, label, full_text_fake_prob, aggregated_fake_prob,
            explanation, top_snippets, raw_output
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        article_id,
        parsed.get("label"),
        parsed.get("full_text_fake_prob"),
        parsed.get("aggregated_fake_prob"),
        parsed.get("explanation"),
        json.dumps(parsed.get("top_snippets", [])),
        raw
    ))
    conn.commit()


# =========================
# Metrics (your request)
# =========================
def compute_metrics(df_pred: pd.DataFrame) -> tuple[float, float]:
    """
    Returns:
      false_positive_rate_on_real, overall_accuracy
    Mapping for correctness:
      actual=real -> predicted REAL is correct
      actual=misinfo -> predicted FAKE is correct
    """
    # FPR on real: predicted FAKE when actual is real
    real = df_pred[df_pred["actual_label"] == "real"]
    if len(real) == 0:
        fpr = float("nan")
    else:
        fpr = (real["predicted_label"] == "FAKE").mean()

    # Overall accuracy under the above mapping
    correct = (
        ((df_pred["actual_label"] == "real") & (df_pred["predicted_label"] == "REAL")) |
        ((df_pred["actual_label"] == "misinfo") & (df_pred["predicted_label"] == "FAKE"))
    )
    acc = correct.mean() if len(df_pred) else float("nan")

    return fpr, acc


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    got_real = collect_from_feeds(conn, "real", REAL_FEEDS, TOTAL_REAL, MAX_CHARS_REAL)
    got_mis  = collect_from_feeds(conn, "misinfo", MISINFO_FEEDS, TOTAL_MISINFO, MAX_CHARS_MISINFO)

    print(f"Collected real: {got_real}/{TOTAL_REAL}, misinfo: {got_mis}/{TOTAL_MISINFO}")

    df_articles = pd.read_sql_query("""
        SELECT id, bucket, source, url, title, published, content
        FROM articles
        ORDER BY id ASC
    """, conn)

    print(f"Running predict.py on {len(df_articles)} rows...")

    for _, row in df_articles.iterrows():
        article_id = int(row["id"])
        text = row["content"]
        parsed, raw = run_predict_py(text)
        save_prediction(conn, article_id, parsed, raw)
        time.sleep(SLEEP_BETWEEN_PREDICTS)

    df_pred = pd.read_sql_query("""
        SELECT
            p.id as prediction_id,
            a.bucket as actual_label,
            p.label as predicted_label,
            a.source,
            a.url,
            a.title,
            a.published,
            p.full_text_fake_prob,
            p.aggregated_fake_prob,
            p.explanation,
            p.top_snippets,
            p.ran_at
        FROM predictions p
        JOIN articles a ON a.id = p.article_id
        ORDER BY p.id DESC
    """, conn)

    df_pred.to_csv(CSV_PATH, index=False)

    fpr, acc = compute_metrics(df_pred)
    print(f"Saved DB file: {DB_PATH}")
    print(f"Saved predictions.csv: {CSV_PATH}")
    print(f"False Positive Rate on REAL news: {fpr:.3f}")
    print(f"Overall accuracy (real->REAL, misinfo->FAKE): {acc:.3f}")

    print(df_pred.head(10).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()