import json
from pathlib import Path

TELUGU_START = 0x0C00
TELUGU_END = 0x0C7F
LENGTH_RATIO_MIN = 0.3
LENGTH_RATIO_MAX = 4.0
MIN_ROMAN_CHARS = 2
MIN_TELUGU_CHARS = 1
REVIEW_BUCKET_THRESHOLD = 0.4


def _telugu_ratio(text: str) -> float:
    if not text:
        return 0.0
    te = sum(1 for c in text if TELUGU_START <= ord(c) <= TELUGU_END)
    return te / len(text)


def _length_ratio(roman: str, telugu: str) -> float:
    if not telugu:
        return 0.0
    return len(roman) / len(telugu)


def rule_score(pair: dict) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 1.0

    roman = pair.get("roman_text") or ""
    telugu = pair.get("telugu_text") or ""

    if len(roman.strip()) < MIN_ROMAN_CHARS:
        return 0.0, ["roman_too_short"]

    if len(telugu.strip()) < MIN_TELUGU_CHARS:
        return 0.0, ["telugu_too_short"]

    te_ratio = _telugu_ratio(telugu)
    if te_ratio < 0.1:
        return 0.0, ["telugu_no_script"]

    ratio = _length_ratio(roman, telugu)
    if ratio < LENGTH_RATIO_MIN or ratio > LENGTH_RATIO_MAX:
        score -= 0.4
        reasons.append(f"length_ratio_out_of_range:{ratio:.2f}")

    # Penalise if roman has Telugu script chars (not properly transliterated)
    if _telugu_ratio(roman) > 0.05:
        score -= 0.3
        reasons.append("roman_has_telugu_chars")

    # Penalise augmented variants slightly
    if pair.get("pair_source") == "augmented":
        score -= 0.05

    # Use existing confidence from upstream as a signal
    upstream_conf = pair.get("confidence", 0.75)
    score = score * 0.6 + upstream_conf * 0.4

    score = max(0.0, min(1.0, round(score, 4)))
    return score, reasons


def score_and_route(pair: dict) -> dict:
    score, reasons = rule_score(pair)
    status = "review" if score < REVIEW_BUCKET_THRESHOLD else "approved"
    return {
        **pair,
        "quality_score": score,
        "score_reasons": reasons,
        "review_status": status,
    }


def score_snapshot(input_path: Path, output_path: Path, review_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    approved = rejected = review_count = 0

    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fapproved, \
         review_path.open("w", encoding="utf-8") as freview:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            scored = score_and_route(pair)

            if scored["quality_score"] == 0.0:
                rejected += 1
            elif scored["review_status"] == "review":
                review_count += 1
                freview.write(json.dumps(scored, ensure_ascii=False) + "\n")
            else:
                approved += 1
                fapproved.write(json.dumps(scored, ensure_ascii=False) + "\n")

    return {
        "stage": "quality_score",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "review_file": str(review_path),
        "approved": approved,
        "review_bucket": review_count,
        "rejected": rejected,
    }


def main() -> None:
    interim_dir = Path("data/interim")
    review_dir = Path("data/review")
    inputs = sorted(interim_dir.glob("deduped_*.jsonl"))

    if not inputs:
        print("No deduped pair files found — run dedup.py first.")
        return

    for inp in inputs:
        out = interim_dir / inp.name.replace("deduped_", "scored_")
        review = review_dir / inp.name.replace("deduped_", "review_")
        print(f"Scoring {inp.name} ...")
        result = score_snapshot(inp, out, review)
        total = result["approved"] + result["review_bucket"] + result["rejected"]
        print(f"  {total:,} pairs -> approved:{result['approved']:,}  "
              f"review:{result['review_bucket']:,}  rejected:{result['rejected']:,}")


if __name__ == "__main__":
    main()
