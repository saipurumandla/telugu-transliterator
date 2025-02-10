import json
import re
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_PATTERN = re.compile(r"^(.+?)_(\d{8})\.jsonl$")


def build_snapshot_index(raw_dir: Path, manifests_dir: Path) -> dict:
    snapshots = []

    for file_path in sorted(raw_dir.glob("*.jsonl")):
        match = SNAPSHOT_PATTERN.match(file_path.name)
        if not match:
            continue

        source_name, pull_date = match.groups()
        size_bytes = file_path.stat().st_size
        record_count = 0
        row_hashes_sample = []

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record_count += 1
                if len(row_hashes_sample) < 5:
                    try:
                        row_hashes_sample.append(json.loads(line).get("row_hash", ""))
                    except json.JSONDecodeError:
                        pass

        snapshots.append({
            "filename": file_path.name,
            "source_name": source_name,
            "pull_date": pull_date,
            "record_count": record_count,
            "size_bytes": size_bytes,
            "row_hashes_sample": row_hashes_sample,
        })

    return {
        "last_updated": datetime.now(tz=timezone.utc).isoformat(),
        "snapshots": snapshots,
    }


def save_index(index: dict, manifests_dir: Path) -> None:
    index_file = manifests_dir / "snapshot_index.json"
    index_file.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def load_index(manifests_dir: Path) -> dict:
    index_file = manifests_dir / "snapshot_index.json"
    if not index_file.exists():
        return {"snapshots": []}
    return json.loads(index_file.read_text(encoding="utf-8"))


def main() -> None:
    raw_dir = Path("data/raw")
    manifests_dir = Path("data/manifests")
    manifests_dir.mkdir(parents=True, exist_ok=True)

    index = build_snapshot_index(raw_dir, manifests_dir)
    save_index(index, manifests_dir)
    print(f"Indexed {len(index['snapshots'])} snapshots -> data/manifests/snapshot_index.json")


if __name__ == "__main__":
    main()
