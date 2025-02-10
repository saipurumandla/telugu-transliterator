import csv
import hashlib
import io
import json
import sys
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

TAR_PATH = Path("downloads/dakshina_dataset_v1.0.tar")
TAR_MEMBER = "dakshina_dataset_v1.0/te/lexicons/te.translit.sampled.train.tsv"
SOURCE_URL = "https://github.com/google-research-datasets/dakshina"
LICENSE_TAG = "CC-BY-SA-4.0"
DOWNLOAD_URL = "https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar"


def _row(text: str, script_hint: str, doc_id: str, pull_ts: str) -> dict:
    return {
        "source_name": "dakshina",
        "source_doc_id": doc_id,
        "source_url": SOURCE_URL,
        "license_tag": LICENSE_TAG,
        "pull_timestamp_utc": pull_ts,
        "text_raw": text,
        "script_hint": script_hint,
        "lang_hint": "te",
        "row_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    if not TAR_PATH.exists():
        print(
            f"Tar not found at {TAR_PATH}\n"
            f"Download (~1.9GB): {DOWNLOAD_URL}\n"
            f"Place it at: {TAR_PATH}"
        )
        sys.exit(1)

    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    pull_ts = datetime.now(tz=timezone.utc).isoformat()
    output_file = Path(f"data/raw/dakshina_{today}.jsonl")
    manifest_file = Path(f"data/manifests/dakshina_{today}_manifest.json")

    if output_file.exists():
        print(f"Snapshot {output_file} already exists — raw/ is immutable, skipping.")
        sys.exit(0)

    print(f"Extracting Telugu lexicon from {TAR_PATH} ...")
    with tarfile.open(TAR_PATH, "r") as tar:
        member = tar.extractfile(TAR_MEMBER)
        if member is None:
            print(f"ERROR: {TAR_MEMBER} not found in tar.")
            sys.exit(1)
        content = member.read().decode("utf-8")

    records = []
    reader = csv.reader(io.StringIO(content), delimiter="\t")
    for line_num, parts in enumerate(reader, start=1):
        if not parts or parts[0].startswith("#"):
            continue
        telugu_word = parts[0].strip()
        if not telugu_word:
            continue

        records.append(_row(telugu_word, "telugu", f"te_lex_{line_num}_0", pull_ts))

        for var_idx, roman in enumerate(parts[1:], start=1):
            roman = roman.strip()
            if roman:
                records.append(_row(roman, "roman", f"te_lex_{line_num}_{var_idx}", pull_ts))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": str(uuid.uuid4()),
        "source_name": "dakshina",
        "pull_timestamp_utc": pull_ts,
        "record_count": len(records),
        "output_file": str(output_file),
        "license_tag": LICENSE_TAG,
    }
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} records -> {output_file}")
    print(f"Manifest -> {manifest_file}")


if __name__ == "__main__":
    main()
