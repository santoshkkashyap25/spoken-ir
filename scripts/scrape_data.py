"""Scrape stand-up transcripts from scrapsfromtheloft.com.

Resumable: per-transcript .json files in data/raw/transcripts/. Re-running
this script only fetches what's missing. After scraping, combines raw
paragraphs into a Transcript string.

Stage 2 (transcript downloads) uses a ThreadPoolExecutor to fetch
multiple transcripts in parallel, significantly reducing total runtime.
"""

from __future__ import annotations

import logging
import os
import json
import re
import string
import sys
import threading
import time
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect

from config import SCRAPING_BASE_URL, RAW_DATA_DIR, TRANSCRIPTS_RAW_DIR

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scrape_data")

# Number of parallel workers for transcript downloads.
_MAX_WORKERS: int = 4

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

# Thread-local storage: each thread gets its own requests.Session so
# connections are not shared across threads (avoids contention).
_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return the per-thread requests.Session, creating one if needed."""
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": _USER_AGENT})
        _thread_local.session = s
    return _thread_local.session


def _fetch_html(url: str, retries: int = 7, backoff_factor: float = 3.0,
                max_backoff: float = 120.0) -> str | None:
    """Fetch a URL with exponential-backoff retries (thread-safe)."""
    # Politeness delay: 3–6 s between every request.
    time.sleep(random.uniform(3.0, 6.0))
    session = _get_session()
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=20)
            if response.status_code == 429:
                # Honour the Retry-After header if the server sends one.
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_time = int(retry_after) + random.uniform(1, 3)
                else:
                    sleep_time = min(backoff_factor * (2 ** attempt), max_backoff)
                logger.warning(
                    "429 Too Many Requests for %s. Retrying in %.1f s (attempt %d/%d)…",
                    url, sleep_time, attempt + 1, retries,
                )
                time.sleep(sleep_time)
                continue
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                logger.error("Error fetching URL %s: %s", url, e)
                return None
            sleep_time = min(backoff_factor * (2 ** attempt), max_backoff)
            logger.warning(
                "Error fetching %s (%s). Retrying in %.1f s (attempt %d/%d)…",
                url, e, sleep_time, attempt + 1, retries,
            )
            time.sleep(sleep_time)
    return None


# Letters the site uses for its A-Z filter.
_LETTERS: list[str] = ["#"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def _is_transcript_url(href: str) -> bool:
    """Return True if the URL looks like an individual transcript page."""
    if not href or "scrapsfromtheloft.com" not in href:
        return False
    # Filter out index pages and the su_letter filter links.
    if "su_letter=" in href:
        return False
    if href.rstrip("/") in [
        "https://scrapsfromtheloft.com/comedy",
        "https://scrapsfromtheloft.com/movies",
        "https://scrapsfromtheloft.com/stand-up-comedy-scripts",
    ]:
        return False
    # Transcripts live under /comedy/ or /movies/
    if "/comedy/" in href or "/movies/" in href:
        return True
    return False


def scrape_links_and_tags(base_url: str) -> tuple[list[str], list[str]]:
    """Scrape all transcript links and display titles in a single pass.

    Returns a tuple of (links, tags) in corresponding order.
    """
    seen: set[str] = set()
    links: list[str] = []
    href_to_title: dict[str, str] = {}

    for letter in _LETTERS:
        if letter == "#":
            url = base_url
        else:
            url = f"{base_url}?su_letter={letter}"
        logger.info("Scraping links/tags for letter '%s' from %s", letter, url)
        html = _fetch_html(url)
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if _is_transcript_url(href):
                if text and href not in href_to_title:
                    href_to_title[href] = text
                if href not in seen:
                    seen.add(href)
                    links.append(href)
        logger.info("  → %d unique links so far", len(links))

    logger.info("Total transcript links found: %d", len(links))

    tags: list[str] = []
    for link in links:
        if link in href_to_title:
            tags.append(href_to_title[link])
        else:
            # Derive a readable tag from the URL slug.
            slug = link.rstrip("/").split("/")[-1]
            slug = slug.replace("-transcript", "").replace("-", " ").title()
            tags.append(slug)
    return links, tags


def scrape_transcript(url: str, content_id: int) -> list[str]:
    """Fetch and cache a single transcript as JSON. Cached on disk per content_id."""
    os.makedirs(TRANSCRIPTS_RAW_DIR, exist_ok=True)
    out_path = os.path.join(TRANSCRIPTS_RAW_DIR, f"{content_id}.json")
    if os.path.exists(out_path):
        logger.info("Transcript for %s already exists. Skipping.", content_id)
        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Scraping transcript for content ID %s from %s", content_id, url)
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")

    # Primary: Astra theme content wrapper used on the new site.
    content_el = (
        soup.find(class_="entry-content")
        or soup.find(class_="ast-single-post-content")
    )

    # Fallback 1: old Elementor class (kept for any cached pages that haven't updated).
    if not content_el:
        content_el = soup.find(
            class_="elementor-element elementor-element-74af9a5b elementor-widget "
            "elementor-widget-theme-post-content"
        )

    # Fallback 2: generic article / main element.
    if not content_el:
        content_el = soup.find("article") or soup.find("main") or soup.find(
            class_=re.compile(r"content|post|article")
        )

    if content_el:
        paragraphs = [p.text for p in content_el.find_all("p")]
    else:
        logger.warning("Could not find transcript content for %s", url)
        return []

    if not paragraphs:
        logger.warning("Found content element but no <p> tags for %s", url)
        return []

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(paragraphs, f)
    return paragraphs


def combine_text(paragraphs: list[str]) -> str:
    return " ".join(paragraphs)



def _extract_name(tag: str) -> str:
    if ":" in tag:
        return tag.split(":")[0].strip()
    if "’s" in tag:
        return tag.split("’s")[0].strip() + "’s"
    if "," in tag and "(" in tag:
        return tag.split(",")[-1].split("(")[0].strip()
    return tag.split("(")[0].strip()


def _extract_title(row: pd.Series) -> str:
    tag = str(row.get("CleanTag") or "")
    name = str(row.get("Names") or "")
    year = str(row.get("Year") or "")
    title = tag.replace(name, "", 1).replace(year, "", 1)
    return re.sub(r"[():]", "", title).strip(" -")


def scrape_and_clean_data(limit: int | None = None) -> pd.DataFrame:
    """End-to-end scrape + clean pipeline. Resumable at every stage."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPTS_RAW_DIR, exist_ok=True)
    output_csv = os.path.join(RAW_DATA_DIR, "scraped_and_cleaned_content_data.csv")

    df = pd.DataFrame()
    if os.path.exists(output_csv):
        try:
            df = pd.read_csv(output_csv)
            logger.info("Resuming from existing CSV with %d records", len(df))
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load existing CSV; starting fresh: %s", e)
            df = pd.DataFrame()

    # --- Stage 1: links + tags ---
    if df.empty or "URL" not in df.columns or "Tag" not in df.columns:
        links, tags = scrape_links_and_tags(SCRAPING_BASE_URL)
        if not links or not tags:
            logger.error("Failed to scrape links or tags. Aborting.")
            return pd.DataFrame()

        n = min(len(links), len(tags))
        if limit is not None:
            n = min(n, limit)
        links, tags = links[:n], tags[:n]
        df = pd.DataFrame({"Tag": tags, "URL": links})
        df.insert(0, "S No.", np.arange(len(df)))
        df.to_csv(output_csv, index=False)
        logger.info("Initial links/tags saved (%d records)", len(df))

    # --- Stage 2: transcripts (parallel, resumable per file) ---
    # Build list of (url, content_id) pairs that still need fetching.
    to_fetch: list[tuple[str, int]] = []
    for _, row in df.iterrows():
        cid = int(row["S No."])
        json_path = os.path.join(TRANSCRIPTS_RAW_DIR, f"{cid}.json")
        if not os.path.exists(json_path):
            to_fetch.append((row["URL"], cid))

    if to_fetch:
        logger.info(
            "Stage 2: downloading %d transcripts with %d parallel workers",
            len(to_fetch), _MAX_WORKERS,
        )
        done_count = 0
        failed_count = 0
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {
                executor.submit(scrape_transcript, url, cid): cid
                for url, cid in to_fetch
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    result = future.result()
                    if result:
                        done_count += 1
                    else:
                        failed_count += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error("Transcript %s raised %s", cid, exc)
                    failed_count += 1
                # Progress log every 25 transcripts.
                total = done_count + failed_count
                if total % 25 == 0 or total == len(to_fetch):
                    logger.info(
                        "  Progress: %d/%d done (%d succeeded, %d failed)",
                        total, len(to_fetch), done_count, failed_count,
                    )
    else:
        logger.info("Stage 2: all transcripts already cached, nothing to fetch.")

    # Reload all cached json files into the DataFrame.
    raw_transcripts: list[list[str]] = []
    for _, row in df.iterrows():
        cid = int(row["S No."])
        json_path = os.path.join(TRANSCRIPTS_RAW_DIR, f"{cid}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                raw_transcripts.append(json.load(f))
        else:
            raw_transcripts.append([])
    df["Raw Transcript"] = raw_transcripts
    df.to_csv(output_csv, index=False)

    df = df[df["Raw Transcript"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    df = df.reset_index(drop=True)
    logger.info("After transcript pass: %d records", len(df))

    # --- Stage 3: combine ---
    if "Transcript" not in df.columns:
        df["Transcript"] = df["Raw Transcript"].apply(combine_text)
        df.to_csv(output_csv, index=False)
    
    df = df[df["Transcript"].str.strip() != ""]
    df = df.reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    logger.info("After combine: %d records", len(df))

    # --- Stage 4: extract Names / Title / Year from Tag ---
    if (
        "Names" not in df.columns
        or "Title" not in df.columns
        or "Year" not in df.columns
    ):
        df["CleanTag"] = df["Tag"].str.split("|").str[0].str.strip()
        df["Year"] = df["CleanTag"].str.extract(r"(\d{4})")
        df["Names"] = df["CleanTag"].apply(_extract_name)
        df["Title"] = df.apply(_extract_title, axis=1)
        df.to_csv(output_csv, index=False)
        logger.info("Extracted Names/Title/Year: %d records", len(df))

    # --- Stage 5: language detection ---
    if "language" not in df.columns or df["language"].isnull().any():
        if "language" not in df.columns:
            df["language"] = np.nan
        for index, row in df.iterrows():
            if pd.isna(row["language"]) or str(row["language"]).strip() == "":
                try:
                    text = str(row["Transcript"])
                    df.at[index, "language"] = detect(text[:500]) if text.strip() else np.nan
                except Exception:  # noqa: BLE001
                    df.at[index, "language"] = np.nan
        df.to_csv(output_csv, index=False)
        logger.info("Language detection complete")

    return df.replace(r"^\s*$", np.nan, regex=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape and clean transcript data.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of links to scrape.")
    args = parser.parse_args()
    
    result = scrape_and_clean_data(limit=args.limit)
    logger.info("Scraping complete: %d records", len(result))
    if "language" in result.columns:
        logger.info("Language breakdown: %s", result["language"].value_counts().to_dict())
    if "Year" in result.columns:
        logger.info("Specials missing Year: %d", result["Year"].isna().sum())
    if "rating" in result.columns:
        logger.info("Specials missing rating: %d", result["rating"].isna().sum())
