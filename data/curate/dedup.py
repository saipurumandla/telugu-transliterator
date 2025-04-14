import hashlib
import json
from pathlib import Path

from datasketch import MinHash, MinHashLSH

NUM_PERM = 128
NEAR_DUP_THRESHOLD = 0.85
SHINGLE_SIZE = 3


def _pair_key(pair: dict) -> str:
    te = (pair.get("telugu_text") or "").strip().lower()
    ro = (pair.get("roman_text") or "").strip().lower()
    return hashlib.sha256(f"{te}|||{ro}".encode("utf-8")).hexdigest()


def _make_minhash(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    tokens = [text[i:i+SHINGLE_SIZE] for i in range(len(text) - SHINGLE_SIZE + 1)]
    for token in tokens:
        m.update(token.encode("utf-8"))
    return m


def dedup_snapshot(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1 — exact dedup
    seen_exact: set[str] = set()
    exact_dupes = 0
    pass1: list[dict] = []

    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            key = _pair_key(pair)
            if key in seen_exact:
                exact_dupes += 1
            else:
                seen_exact.add(key)
                pass1.append(pair)

    # Pass 2 — near dedup on roman_text using MinHash LSH
    lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=NUM_PERM)
    kept: list[dict] = []
    near_dupes = 0

    for i, pair in enumerate(pass1):
        roman = (pair.get("roman_text") or "").strip().lower()
        if len(roman) < SHINGLE_SIZE:
            kept.append(pair)
            continue

        m = _make_minhash(roman)
        candidates = lsh.query(m)
        if candidates:
            near_dupes += 1
        else:
            lsh.insert(str(i), m)
            kept.append(pair)

    with output_path.open("w", encoding="utf-8") as f:
        for pair in kept:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return {
        "stage": "dedup",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_count": len(pass1) + exact_dupes,
        "exact_dupes": exact_dupes,
        "near_dupes": near_dupes,
        "output_count": len(kept),
    }


def main() -> None:
    interim_dir = Path("data/interim")
    inputs = sorted(interim_dir.glob("augmented_*.jsonl"))

    if not inputs:
        print("No augmented pair files found in data/interim/")
        return

    for inp in inputs:
        out = interim_dir / inp.name.replace("augmented_", "deduped_")
        print(f"Deduping {inp.name} ...")
        result = dedup_snapshot(inp, out)
        removed = result["exact_dupes"] + result["near_dupes"]
        print(f"  {result['input_count']:,} in -> {result['output_count']:,} kept")
        print(f"    exact dupes: {result['exact_dupes']:,}  near dupes: {result['near_dupes']:,}")


if __name__ == "__main__":
    main()
