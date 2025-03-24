import json
import re
import unicodedata
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Long vowel markers in ITRANS → simplified (drop length distinction)
_SIMPLIFY = str.maketrans({
    "A": "a", "I": "i", "U": "u",
    "è": "e", "ò": "o",
    "M": "m", "H": "h",
    "~": "",
})

_MULTI_SPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_SIMPLIFY)
    text = _MULTI_SPACE.sub(" ", text).strip().lower()
    return text


def romanize(telugu_text: str) -> str:
    raw = transliterate(telugu_text, sanscript.TELUGU, sanscript.ITRANS)
    return _clean(raw)


def romanize_variants(telugu_text: str) -> list[str]:
    itrans_raw = transliterate(telugu_text, sanscript.TELUGU, sanscript.ITRANS)
    simplified = _clean(itrans_raw)

    # Velthuis doubles vowels for length (e.g. vastunnaanu) — another common style
    velthuis_raw = transliterate(telugu_text, sanscript.TELUGU, sanscript.VELTHUIS)
    velthuis = _multi_space_clean(velthuis_raw.lower())

    variants = [simplified]
    if velthuis != simplified:
        variants.append(velthuis)
    return variants


def _multi_space_clean(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text).strip()


def fill_pending_pairs(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = filled = skipped = 0

    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            pair = json.loads(line)

            if pair.get("pair_source") == "wikipedia_pending":
                telugu = pair.get("telugu_text", "")
                if not telugu:
                    skipped += 1
                    continue
                roman = romanize(telugu)
                if not roman:
                    skipped += 1
                    continue
                pair["roman_text"] = roman
                pair["pair_source"] = "synthetic"
                pair["quality_score"] = 0.70
                pair["confidence"] = 0.70
                pair["review_status"] = "approved"
                filled += 1

            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return {
        "stage": "romanize_rules",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_pairs": total,
        "filled": filled,
        "skipped": skipped,
    }


def main() -> None:
    interim_dir = Path("data/interim")
    inputs = sorted(interim_dir.glob("pairs_*.jsonl"))

    if not inputs:
        print("No pair files found in data/interim/")
        return

    for inp in inputs:
        out = interim_dir / inp.name.replace("pairs_", "romanized_")
        print(f"Filling roman forms in {inp.name} ...")
        result = fill_pending_pairs(inp, out)
        print(f"  {result['total_pairs']:,} pairs — {result['filled']:,} filled, {result['skipped']:,} skipped")


if __name__ == "__main__":
    main()
