"""Scrape stand-up transcripts from scrapsfromtheloft.com.

Resumable: per-transcript .pkl files in data/raw/transcripts/. Re-running
this script only fetches what's missing. After scraping, combines raw
paragraphs into a Transcript string and runs a regex-based clean.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import string
import sys
import warnings

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect

# Allow running directly from any working directory: `python scripts/scrape_data.py`
# scripts/scrape_data.py → scripts/ → project root.
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from config import SCRAPING_BASE_URL, RAW_DATA_DIR, TRANSCRIPTS_RAW_DIR  # noqa: E402

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("scrape_data")


def _fetch_html(url: str) -> str | None:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching URL %s: %s", url, e)
        return None


def scrape_links(url: str) -> list[str]:
    logger.info("Scraping content links from %s", url)
    html = _fetch_html(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    section = soup.find(
        class_="elementor-section elementor-top-section elementor-element "
        "elementor-element-b70b8d7 elementor-section-boxed elementor-section-height-default "
        "elementor-section-height-default"
    )
    if section:
        return [a.get("href") for a in section.find_all("a")]
    logger.warning("Specific elementor-section class not found; falling back to broader search.")
    return [
        a.get("href")
        for a in soup.find_all("a", href=True)
        if SCRAPING_BASE_URL in a.get("href")
        and "/stand-up-comedy-scripts/" in a.get("href")
        and a.get("href") != SCRAPING_BASE_URL
    ]


def scrape_tags(url: str) -> list[str]:
    logger.info("Scraping tags (titles) from %s", url)
    html = _fetch_html(url)
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    section = soup.find(
        class_="elementor-section elementor-top-section elementor-element "
        "elementor-element-b70b8d7 elementor-section-boxed elementor-section-height-default "
        "elementor-section-height-default"
    )
    if section:
        return [h.text.strip() for h in section.find_all("h3")]
    logger.warning("Specific elementor-section class not found; falling back to broader search.")
    return [h.text.strip() for h in soup.find_all("h3")]


def scrape_transcript(url: str, content_id: int) -> list[str]:
    """Fetch and pickle a single transcript. Cached on disk per content_id."""
    os.makedirs(TRANSCRIPTS_RAW_DIR, exist_ok=True)
    out_path = os.path.join(TRANSCRIPTS_RAW_DIR, f"{content_id}.pkl")
    if os.path.exists(out_path):
        logger.info("Transcript for %s already exists. Skipping.", content_id)
        with open(out_path, "rb") as f:
            return pickle.load(f)

    logger.info("Scraping transcript for content ID %s from %s", content_id, url)
    html = _fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    content_el = soup.find(
        class_="elementor-element elementor-element-74af9a5b elementor-widget elementor-widget-theme-post-content"
    )
    if content_el:
        paragraphs = [p.text for p in content_el.find_all("p")]
    else:
        # Fallback: find the main content area.
        main = soup.find("article") or soup.find("main") or soup.find(
            class_=re.compile(r"content|post|article")
        )
        paragraphs = [p.text for p in main.find_all("p")] if main else []
        if not paragraphs:
            logger.warning("Could not find transcript content for %s", url)
            return []

    with open(out_path, "wb") as f:
        pickle.dump(paragraphs, f)
    return paragraphs


def combine_text(paragraphs: list[str]) -> str:
    return " ".join(paragraphs)


def clean_text_content(text: str) -> str:
    """Light clean: lowercase, strip punctuation, newlines, square brackets, and number-words."""
    text = re.sub(r"\[.*?\]", "", text)
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\n", "", text)
    text = re.sub(r"[‘’“”…♪)(“”…]", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


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
        links = scrape_links(SCRAPING_BASE_URL)
        tags = scrape_tags(SCRAPING_BASE_URL)
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

    # --- Stage 2: transcripts (resumable per file) ---
    if "Raw Transcript" not in df.columns:
        df["Raw Transcript"] = [[]] * len(df)

    for index, row in df.iterrows():
        pkl_path = os.path.join(TRANSCRIPTS_RAW_DIR, f"{row['S No.']}.pkl")
        existing = row.get("Raw Transcript")
        if os.path.exists(pkl_path) and isinstance(existing, list) and len(existing) > 0:
            continue
        transcript = scrape_transcript(row["URL"], row["S No."])
        df.at[index, "Raw Transcript"] = transcript
        df.to_csv(output_csv, index=False)

    df = df[df["Raw Transcript"].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    df = df.reset_index(drop=True)
    logger.info("After transcript pass: %d records", len(df))

    # --- Stage 3: combine + clean ---
    if "Transcript" not in df.columns:
        df["Transcript"] = df["Raw Transcript"].apply(combine_text)
        df.to_csv(output_csv, index=False)
    df["Transcript"] = df["Transcript"].apply(clean_text_content)
    df = df[df["Transcript"].str.strip() != ""]
    df = df.reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    logger.info("After cleaning: %d records", len(df))

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
    result = scrape_and_clean_data()
    logger.info("Scraping complete: %d records", len(result))
