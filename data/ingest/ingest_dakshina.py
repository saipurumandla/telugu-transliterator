import json
import hashlib
import uuid
import datetime
from pathlib import Path
import requests


DAKSHINA_TE_LEXICON_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/dakshina/"
    "master/dakshina_dataset_v1.0/te/lexicon/te.romanized.rejoined.lexicon.tsv"
)
SOURCE_URL = "https://github.com/google-research-datasets/dakshina"
LICENSE_TAG = "CC-BY-SA-4.0"


def _make_record(text: str, script_hint: str, pull_ts: str) -> dict:
    return {
        "source_name": "dakshina",
        "source_doc_id": str(uuid.uuid4()),
        "source_url": SOURCE_URL,
        "license_tag": LICENSE_TAG,
        "pull_timestamp_utc": pull_ts,
        "text_raw": text,
        "script_hint": script_hint,
        "lang_hint": "te",
        "row_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    today = datetime.date.today().strftime("%Y%m%d")
    output_file = Path(f"data/raw/dakshina_{today}.jsonl")
    manifest_file = Path(f"data/manifests/dakshina_{today}_manifest.json")

    if output_file.exists():
        print(f"Snapshot {output_file} already exists — raw/ is immutable, skipping.")
        return

    print(f"Pulling Dakshina Telugu lexicon from GitHub...")
    response = requests.get(DAKSHINA_TE_LEXICON_URL, timeout=60)
    response.raise_for_status()

    pull_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []

    for line in response.text.strip().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        native_text, roman_text = parts[0].strip(), parts[1].strip()
        if not native_text or not roman_text:
            continue
        records.append(_make_record(native_text, "telugu", pull_ts))
        records.append(_make_record(roman_text, "roman", pull_ts))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": str(uuid.uuid4()),
        "source_name": "dakshina",
        "pull_date": today,
        "record_count": len(records),
        "output_file": str(output_file),
    }
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {output_file}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
