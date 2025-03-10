import json
import re
from pathlib import Path

MIN_CHARS = 2
MAX_CHARS = 512
MAX_DIGIT_RATIO = 0.6
MAX_REPEAT_RATIO = 0.6  # single char repeated > 60% of string = noise

URL_PATTERN = re.compile(
    r'^(https?://|www\.)\S+$', re.IGNORECASE
)
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]+",
    re.UNICODE,
)


def is_url_only(text: str) -> bool:
    return bool(URL_PATTERN.match(text.strip()))


def is_emoji_only(text: str) -> bool:
    stripped = EMOJI_PATTERN.sub("", text).strip()
    return len(stripped) == 0 and len(text) > 0


def is_too_short(text: str) -> bool:
    return len(text.strip()) < MIN_CHARS


def is_too_long(text: str) -> bool:
    return len(text) > MAX_CHARS


def is_high_digit_ratio(text: str) -> bool:
    if not text:
        return False
    digit_count = sum(1 for c in text if c.isdigit())
    return digit_count / len(text) > MAX_DIGIT_RATIO


def is_high_repeat(text: str) -> bool:
    if len(text) < 4:
        return False
    most_common = max(set(text), key=text.count)
    return text.count(most_common) / len(text) > MAX_REPEAT_RATIO


def check_record(record: dict) -> str | None:
    text = record.get("text_normalized") or record.get("text_raw", "")
    if is_too_short(text):
        return "too_short"
    if is_too_long(text):
        return "too_long"
    if is_url_only(text):
        return "url_only"
    if is_emoji_only(text):
        return "emoji_only"
    if is_high_digit_ratio(text):
        return "high_digit_ratio"
    if is_high_repeat(text):
        return "high_repeat"
    return None


def filter_snapshot(input_path: Path, output_path: Path) -> dict:
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
            record = json.loads(line)
            reason = check_record(record)
            if reason:
                dropped += 1
                drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            else:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    return {
        "stage": "filter_junk",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "input_count": total,
        "output_count": kept,
        "drop_count": dropped,
        "drop_reasons": drop_reasons,
    }


def main() -> None:
    interim_dir = Path("data/interim")
    inputs = sorted(interim_dir.glob("normalized_*.jsonl"))
    if not inputs:
        print("No normalized snapshots found in data/interim/")
        return

    for inp in inputs:
        out = interim_dir / inp.name.replace("normalized_", "filtered_")
        print(f"Filtering {inp.name} ...")
        manifest = filter_snapshot(inp, out)
        print(f"  {manifest['input_count']} in -> {manifest['output_count']} kept, {manifest['drop_count']} dropped")
        for reason, count in manifest["drop_reasons"].items():
            print(f"    {reason}: {count}")


if __name__ == "__main__":
    main()
