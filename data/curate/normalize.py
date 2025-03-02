import json
import re
import unicodedata
from pathlib import Path

TELUGU_BLOCK_START = 0x0C00
TELUGU_BLOCK_END = 0x0C7F

CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
MULTI_WHITESPACE = re.compile(r'\s+')


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_control_chars(text: str) -> str:
    return CONTROL_CHARS.sub("", text)


def normalize_whitespace(text: str) -> str:
    return MULTI_WHITESPACE.sub(" ", text).strip()


def telugu_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    te_chars = sum(1 for c in text if TELUGU_BLOCK_START <= ord(c) <= TELUGU_BLOCK_END)
    return te_chars / len(text)


def detect_script(text: str) -> str:
    ratio = telugu_char_ratio(text)
    if ratio >= 0.4:
        return "telugu"
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    if ascii_letters / max(len(text), 1) >= 0.4:
        return "roman"
    return "mixed"


def normalize_record(record: dict) -> dict:
    raw = record["text_raw"]
    cleaned = normalize_whitespace(remove_control_chars(nfc(raw)))
    detected = detect_script(cleaned)
    return {
        **record,
        "text_normalized": cleaned,
        "script_detected": detected,
    }


def normalize_snapshot(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = dropped = 0
    drop_reasons: dict[str, int] = {}

    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                drop_reasons["json_decode_error"] = drop_reasons.get("json_decode_error", 0) + 1
                continue

            normalized = normalize_record(record)
            cleaned = normalized["text_normalized"]

            if not cleaned:
                dropped += 1
                drop_reasons["empty_after_normalize"] = drop_reasons.get("empty_after_normalize", 0) + 1
                continue

            # Reject if script_hint says telugu but no Telugu chars found
            if record.get("script_hint") == "telugu" and telugu_char_ratio(cleaned) < 0.1:
                dropped += 1
                drop_reasons["telugu_hint_no_script"] = drop_reasons.get("telugu_hint_no_script", 0) + 1
                continue

            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            kept += 1

    return {
        "stage": "normalize",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_count": total,
        "output_count": kept,
        "drop_count": dropped,
        "drop_reasons": drop_reasons,
    }


def main() -> None:
    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim")

    snapshots = sorted(raw_dir.glob("*.jsonl"))
    if not snapshots:
        print("No snapshots found in data/raw/")
        return

    for snap in snapshots:
        out = interim_dir / f"normalized_{snap.name}"
        print(f"Normalizing {snap.name} ...")
        manifest = normalize_snapshot(snap, out)
        print(f"  {manifest['input_count']} in -> {manifest['output_count']} kept, {manifest['drop_count']} dropped")
        if manifest["drop_reasons"]:
            for reason, count in manifest["drop_reasons"].items():
                print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
