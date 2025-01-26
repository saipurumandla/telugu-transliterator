import csv
import json
import os
import sys
import uuid
from datetime import date, datetime, timezone
from hashlib import sha256

OUTPUT_DIR = "data/raw"
MANIFEST_DIR = "data/manifests"
INPUT_FILE = "data/interim/licensed_corpus_raw.tsv"
SOURCE_URL = "https://tdil-dc.in/"
TAKEDOWN_CONTACT = "contact@tdil-dc.in"


def _row_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    if not os.path.exists(INPUT_FILE):
        print(
            f"File not found: {INPUT_FILE}\n"
            "The ILCI corpus requires manual access approval.\n"
            "Download the Telugu parallel data from https://tdil-dc.in/ and place it at that path."
        )
        sys.exit(1)

    today = date.today().strftime("%Y%m%d")
    pull_ts = datetime.now(tz=timezone.utc).isoformat()
    output_file = os.path.join(OUTPUT_DIR, f"ilci_corpus_{today}.jsonl")
    manifest_file = os.path.join(MANIFEST_DIR, f"ilci_corpus_{today}_manifest.json")

    if os.path.exists(output_file):
        print(f"Snapshot {output_file} already exists — raw/ is immutable, skipping.")
        sys.exit(0)

    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            doc_id = row.get("doc_id", "")
            roman = row.get("roman_text", "").strip()
            telugu = row.get("telugu_text", "").strip()
            if not roman or not telugu:
                continue

            records.append({
                "source_name": "ilci_corpus",
                "source_doc_id": doc_id,
                "source_url": SOURCE_URL,
                "license_tag": "CC-BY-NC-4.0",
                "pull_timestamp_utc": pull_ts,
                "text_raw": roman,
                "script_hint": "roman",
                "lang_hint": "te",
                "row_hash": _row_hash(roman),
                "takedown_contact": TAKEDOWN_CONTACT,
            })
            records.append({
                "source_name": "ilci_corpus",
                "source_doc_id": doc_id,
                "source_url": SOURCE_URL,
                "license_tag": "CC-BY-NC-4.0",
                "pull_timestamp_utc": pull_ts,
                "text_raw": telugu,
                "script_hint": "telugu",
                "lang_hint": "te",
                "row_hash": _row_hash(telugu),
                "takedown_contact": TAKEDOWN_CONTACT,
            })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "run_id": str(uuid.uuid4()),
        "source_name": "ilci_corpus",
        "pull_timestamp_utc": pull_ts,
        "record_count": len(records),
        "output_file": output_file,
        "license_tag": "CC-BY-NC-4.0",
        "takedown_contact": TAKEDOWN_CONTACT,
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {output_file}")


if __name__ == "__main__":
    main()
