import hashlib
import io
import json
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ZIP_URL = "https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/tel.zip"
SOURCE_URL = "https://huggingface.co/datasets/ai4bharat/Aksharantar"
LICENSE_TAG = "CC-BY-4.0"
SPLITS = ["tel_train.json", "tel_valid.json", "tel_test.json"]


def _row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_record(native: str, roman: str, doc_id: str, source: str, score, pull_ts: str) -> tuple[dict, dict]:
    te_rec = {
        "source_name": "aksharantar",
        "source_doc_id": f"{doc_id}_te",
        "source_url": SOURCE_URL,
        "license_tag": LICENSE_TAG,
        "pull_timestamp_utc": pull_ts,
        "text_raw": native,
        "script_hint": "telugu",
        "lang_hint": "te",
        "row_hash": _row_hash(native),
        "aksharantar_source": source,
        "aksharantar_score": score,
        "paired_with": f"{doc_id}_ro",
    }
    ro_rec = {
        "source_name": "aksharantar",
        "source_doc_id": f"{doc_id}_ro",
        "source_url": SOURCE_URL,
        "license_tag": LICENSE_TAG,
        "pull_timestamp_utc": pull_ts,
        "text_raw": roman,
        "script_hint": "roman",
        "lang_hint": "te",
        "row_hash": _row_hash(roman),
        "aksharantar_source": source,
        "aksharantar_score": score,
        "paired_with": f"{doc_id}_te",
    }
    return te_rec, ro_rec


def main() -> None:
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    pull_ts = datetime.now(tz=timezone.utc).isoformat()
    output_file = Path(f"data/raw/aksharantar_{today}.jsonl")
    manifest_file = Path(f"data/manifests/aksharantar_{today}_manifest.json")

    if output_file.exists():
        print(f"Snapshot {output_file} already exists -- raw/ is immutable, skipping.")
        sys.exit(0)

    print(f"Downloading Aksharantar tel.zip (~69MB) from HuggingFace ...")
    response = requests.get(ZIP_URL, timeout=180, stream=True)
    response.raise_for_status()

    chunks = []
    downloaded = 0
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        downloaded += len(chunk)
        if downloaded % (10 * 1024 * 1024) == 0:
            print(f"  {downloaded // 1024 // 1024}MB downloaded ...")
    zip_data = b"".join(chunks)
    print(f"Download complete: {len(zip_data) // 1024 // 1024}MB")

    zf = zipfile.ZipFile(io.BytesIO(zip_data))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    split_counts: dict[str, int] = {}

    with output_file.open("w", encoding="utf-8") as fout:
        for split_file in SPLITS:
            if split_file not in zf.namelist():
                print(f"  Warning: {split_file} not found in zip, skipping.")
                continue

            split_name = split_file.replace("tel_", "").replace(".json", "")
            print(f"  Processing {split_file} ...")

            with zf.open(split_file) as f:
                lines = f.read().decode("utf-8").strip().splitlines()

            split_count = 0
            for line in lines:
                if not line.strip():
                    continue
                entry = json.loads(line)
                native = entry.get("native word", "").strip()
                roman = entry.get("english word", "").strip()
                if not native or not roman:
                    continue

                doc_id = entry.get("unique_identifier", str(uuid.uuid4()))
                source = entry.get("source", "unknown")
                score = entry.get("score")

                te_rec, ro_rec = _make_record(native, roman, doc_id, source, score, pull_ts)
                fout.write(json.dumps(te_rec, ensure_ascii=False) + "\n")
                fout.write(json.dumps(ro_rec, ensure_ascii=False) + "\n")
                split_count += 2

            split_counts[split_name] = split_count
            total_records += split_count
            print(f"    {split_count // 2} pairs -> {split_count} records")

    manifest = {
        "run_id": str(uuid.uuid4()),
        "source_name": "aksharantar",
        "pull_timestamp_utc": pull_ts,
        "record_count": total_records,
        "output_file": str(output_file),
        "license_tag": LICENSE_TAG,
        "split_counts": split_counts,
    }
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Done: {total_records} records -> {output_file}")


if __name__ == "__main__":
    main()
