import hashlib
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

# HuggingFace datasets-server pagination API
DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"
DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.te"
SPLIT = "train"
PAGE_SIZE = 100
SOURCE_URL = "https://huggingface.co/datasets/wikimedia/wikipedia"
LICENSE_TAG = "CC-BY-SA-3.0"

# Sentence splitting — split on Telugu danda (।), newline, or double newline
SENTENCE_SPLIT = re.compile(r'[\n।]+')
MIN_SENTENCE_CHARS = 20
MAX_SENTENCE_CHARS = 400

# Telugu Unicode block
TELUGU_START = 0x0C00
TELUGU_END = 0x0C7F


def _is_mostly_telugu(text: str, threshold: float = 0.4) -> bool:
    if not text:
        return False
    te = sum(1 for c in text if TELUGU_START <= ord(c) <= TELUGU_END)
    return te / len(text) >= threshold


def _row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_sentences(article_text: str) -> list[str]:
    sentences = []
    for sent in SENTENCE_SPLIT.split(article_text):
        sent = sent.strip()
        if MIN_SENTENCE_CHARS <= len(sent) <= MAX_SENTENCE_CHARS and _is_mostly_telugu(sent):
            sentences.append(sent)
    return sentences


def _get_total_rows() -> int:
    r = requests.get(
        f"https://datasets-server.huggingface.co/size",
        params={"dataset": DATASET, "config": CONFIG, "split": SPLIT},
        timeout=15,
    )
    if r.status_code == 200:
        return r.json().get("size", {}).get("num_rows", 0)
    return 0


def main() -> None:
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    pull_ts = datetime.now(tz=timezone.utc).isoformat()
    output_file = Path(f"data/raw/wikipedia_te_{today}.jsonl")
    manifest_file = Path(f"data/manifests/wikipedia_te_{today}_manifest.json")

    if output_file.exists():
        print(f"Snapshot {output_file} already exists -- raw/ is immutable, skipping.")
        sys.exit(0)

    total_rows = _get_total_rows()
    print(f"Telugu Wikipedia: {total_rows} articles to process")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    articles_processed = 0
    sentences_written = 0
    offset = 0

    with output_file.open("w", encoding="utf-8") as fout:
        while True:
            params = {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "limit": PAGE_SIZE,
            }
            r = requests.get(DATASETS_SERVER, params=params, timeout=30)
            if r.status_code == 429:
                print(f"Rate limited at offset {offset} -- saving progress and stopping.")
                break
            if r.status_code != 200:
                print(f"API error at offset {offset}: {r.status_code} {r.text[:100]}")
                break

            data = r.json()
            rows = data.get("rows", [])
            if not rows:
                break

            for row_wrapper in rows:
                row = row_wrapper.get("row", {})
                article_id = str(row.get("id", uuid.uuid4()))
                article_url = row.get("url", SOURCE_URL)
                title = row.get("title", "")
                text = row.get("text", "")

                sentences = _extract_sentences(text)
                for sent_idx, sent in enumerate(sentences):
                    record = {
                        "source_name": "wikipedia_te",
                        "source_doc_id": f"wiki_{article_id}_s{sent_idx}",
                        "source_url": article_url,
                        "license_tag": LICENSE_TAG,
                        "pull_timestamp_utc": pull_ts,
                        "text_raw": sent,
                        "script_hint": "telugu",
                        "lang_hint": "te",
                        "row_hash": _row_hash(sent),
                        "article_title": title,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sentences_written += 1

                articles_processed += 1

            offset += PAGE_SIZE
            time.sleep(0.5)
            if offset % 1000 == 0:
                print(f"  {articles_processed} articles, {sentences_written} sentences ...")

            if not data.get("rows") or len(rows) < PAGE_SIZE:
                break

    manifest = {
        "run_id": str(uuid.uuid4()),
        "source_name": "wikipedia_te",
        "pull_timestamp_utc": pull_ts,
        "record_count": sentences_written,
        "output_file": str(output_file),
        "license_tag": LICENSE_TAG,
        "articles_processed": articles_processed,
        "note": "Telugu script only. Roman pairs generated in Phase 2 by romanize_rules.py.",
    }
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Done: {articles_processed} articles -> {sentences_written} sentences -> {output_file}")


if __name__ == "__main__":
    main()
