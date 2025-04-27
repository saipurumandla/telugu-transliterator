import hashlib
import json
import random
from pathlib import Path

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42


def _split_key(pair: dict) -> str:
    # Group near-duplicates by telugu_text hash so they land in same split
    telugu = (pair.get("telugu_text") or "").strip().lower()
    return hashlib.md5(telugu.encode("utf-8")).hexdigest()


def assign_split(key: str) -> str:
    bucket = int(key[:4], 16) / 0xFFFF
    if bucket < TRAIN_RATIO:
        return "train"
    elif bucket < TRAIN_RATIO + VAL_RATIO:
        return "val"
    else:
        return "test"


def split_files(
    input_paths: list[Path],
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for inp in input_paths:
        with inp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                # Only include approved pairs in the split
                if pair.get("review_status") != "approved":
                    continue
                key = _split_key(pair)
                split = assign_split(key)
                splits[split].append(pair)

    random.seed(SEED)
    for split_name, pairs in splits.items():
        random.shuffle(pairs)
        out_path = output_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    counts = {k: len(v) for k, v in splits.items()}

    # Leakage check — no telugu_text should appear in more than one split
    leakage = _check_leakage(splits)

    return {
        "stage": "split_dataset",
        "output_dir": str(output_dir),
        "counts": counts,
        "total": sum(counts.values()),
        "leakage_violations": leakage,
    }


def _check_leakage(splits: dict[str, list[dict]]) -> list[str]:
    violations = []
    split_keys: dict[str, set[str]] = {}
    for name, pairs in splits.items():
        split_keys[name] = {_split_key(p) for p in pairs}

    split_names = list(split_keys.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = split_keys[a] & split_keys[b]
            if overlap:
                violations.append(f"{a}/{b} overlap: {len(overlap)} keys")
    return violations


def main() -> None:
    # Use scored output if available, else fall back to deduped
    interim = Path("data/interim")
    processed = Path("data/processed")

    scored = sorted(interim.glob("scored_*.jsonl"))
    deduped = sorted(interim.glob("deduped_*.jsonl"))
    inputs = scored if scored else deduped

    if not inputs:
        print("No scored or deduped files found — run quality_score.py or dedup.py first.")
        return

    print(f"Splitting {len(inputs)} file(s) -> data/processed/ ...")
    result = split_files(inputs, processed)
    print(f"  train: {result['counts']['train']:,}")
    print(f"  val  : {result['counts']['val']:,}")
    print(f"  test : {result['counts']['test']:,}")
    print(f"  total: {result['total']:,}")

    if result["leakage_violations"]:
        print(f"  WARNING — leakage detected: {result['leakage_violations']}")
    else:
        print("  Leakage check: clean")


if __name__ == "__main__":
    main()
