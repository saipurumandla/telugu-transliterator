import json
import re
import uuid
from pathlib import Path

# Each rule is (pattern, list_of_replacements)
# Applied to the roman side only — produces colloquial chat-style variants
VARIANT_RULES: list[tuple[str, list[str]]] = [
    # Long vowels often shortened in chat
    (r"aa", ["a"]),
    (r"ee", ["e", "i"]),
    (r"oo", ["o", "u"]),
    (r"uu", ["u"]),
    # th/dh are often dropped to t/d
    (r"th", ["t"]),
    (r"dh", ["d"]),
    # v/w interchange is very common in Telugu chat
    (r"\bv", ["w"]),
    (r"\bw", ["v"]),
    # gaa/ga ending variations
    (r"gaa\b", ["ga"]),
    # ra/r at end of words
    (r"ra\b", ["r"]),
    # Final 'u' often dropped in casual speech
    (r"([bcdfghjklmnpqrstvwxyz])u\b", [r"\1"]),
    # Nasal m/n interchange before consonants
    (r"am([bmp])", [r"an\1"]),
    (r"an([bmp])", [r"am\1"]),
    # Phase 4 additions — expanded chat patterns from corpus analysis
    # 'nu' suffix often shortened
    (r"nu\b", ["n"]),
    # 'indi' → 'undi' (very common tense marker variation)
    (r"indi\b", ["undi"]),
    (r"undi\b", ["indi"]),
    # 'ante' → 'ante'/'nte' shortening
    (r"\bante\b", ["nte"]),
    # Double consonant simplification (mm→m, nn→n, ll→l)
    (r"([mnl])\1", [r"\1"]),
    # 'ku' → 'ki' (dative case variation)
    (r"ku\b", ["ki"]),
    # 'lo' → 'la' (locative variation)
    (r"\blo\b", ["la"]),
    # 'ki' → 'ke' (common in fast speech)
    (r"\bki\b", ["ke"]),
]

_COMPILED = [(re.compile(p), reps) for p, reps in VARIANT_RULES]


def generate_variants(roman: str, max_variants: int = 3) -> list[str]:
    variants: set[str] = set()
    for pattern, replacements in _COMPILED:
        if pattern.search(roman):
            for rep in replacements:
                variant = pattern.sub(rep, roman, count=1)
                if variant != roman and variant.strip():
                    variants.add(variant.strip())
        if len(variants) >= max_variants:
            break
    return list(variants)[:max_variants]


def augment_snapshot(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = augmented = skipped = 0

    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            pair = json.loads(line)
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")

            roman = pair.get("roman_text")
            telugu = pair.get("telugu_text")
            if not roman or not telugu:
                skipped += 1
                continue

            # Only augment approved high-confidence pairs
            if pair.get("review_status") != "approved" or pair.get("confidence", 0) < 0.6:
                continue

            variants = generate_variants(roman)
            for variant in variants:
                if variant == roman:
                    continue
                new_pair = {
                    **pair,
                    "pair_id": str(uuid.uuid4()),
                    "roman_text": variant,
                    "pair_source": "augmented",
                    "quality_score": round(pair.get("quality_score", 0.7) * 0.9, 4),
                    "confidence": round(pair.get("confidence", 0.7) * 0.9, 4),
                    "review_status": "approved",
                    "augmentation_variant": roman,
                }
                fout.write(json.dumps(new_pair, ensure_ascii=False) + "\n")
                augmented += 1

    return {
        "stage": "augment_colloquial",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_pairs": total,
        "augmented_pairs": augmented,
        "skipped": skipped,
    }


def main() -> None:
    interim_dir = Path("data/interim")
    inputs = sorted(interim_dir.glob("romanized_*.jsonl"))

    if not inputs:
        print("No romanized pair files found in data/interim/")
        return

    for inp in inputs:
        out = interim_dir / inp.name.replace("romanized_", "augmented_")
        print(f"Augmenting {inp.name} ...")
        result = augment_snapshot(inp, out)
        total_out = result["original_pairs"] + result["augmented_pairs"]
        print(f"  {result['original_pairs']:,} original + {result['augmented_pairs']:,} augmented = {total_out:,} total")


if __name__ == "__main__":
    main()
