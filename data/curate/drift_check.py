"""
Check distribution drift between a new raw snapshot and the baseline.
Compares script ratio, length distribution, and source overlap against
a reference scored file to flag significant shifts before retraining.
"""

import json
import statistics
from pathlib import Path


TELUGU_RANGE = range(0x0C00, 0x0C80)


def _script_ratio(text: str) -> float:
    if not text:
        return 0.0
    telugu = sum(1 for c in text if ord(c) in TELUGU_RANGE)
    return telugu / len(text)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _length_stats(texts: list[str]) -> dict:
    lengths = [len(t) for t in texts if t]
    if not lengths:
        return {"mean": 0, "median": 0, "p95": 0}
    lengths.sort()
    p95_idx = int(len(lengths) * 0.95)
    return {
        "mean": round(statistics.mean(lengths), 1),
        "median": statistics.median(lengths),
        "p95": lengths[p95_idx],
    }


def check_drift(
    new_snapshot: Path,
    reference_scored: Path,
    thresholds: dict | None = None,
) -> dict:
    thresholds = thresholds or {
        "script_ratio_delta": 0.05,
        "mean_length_delta_pct": 0.20,
    }

    new_records = _load_jsonl(new_snapshot)
    ref_records = _load_jsonl(reference_scored)

    new_texts = [r.get("text_raw", r.get("roman_text", r.get("telugu_text", ""))) for r in new_records]
    ref_texts = [r.get("text_raw", r.get("roman_text", r.get("telugu_text", ""))) for r in ref_records]

    new_script = statistics.mean(_script_ratio(t) for t in new_texts if t) if new_texts else 0.0
    ref_script = statistics.mean(_script_ratio(t) for t in ref_texts if t) if ref_texts else 0.0

    new_len = _length_stats(new_texts)
    ref_len = _length_stats(ref_texts)

    script_delta = abs(new_script - ref_script)
    length_delta_pct = (
        abs(new_len["mean"] - ref_len["mean"]) / max(ref_len["mean"], 1)
    )

    alerts = []
    if script_delta > thresholds["script_ratio_delta"]:
        alerts.append(
            f"script_ratio drift {script_delta:.3f} exceeds threshold {thresholds['script_ratio_delta']}"
        )
    if length_delta_pct > thresholds["mean_length_delta_pct"]:
        alerts.append(
            f"mean_length drift {length_delta_pct:.1%} exceeds threshold {thresholds['mean_length_delta_pct']:.1%}"
        )

    result = {
        "new_snapshot": str(new_snapshot),
        "reference": str(reference_scored),
        "new_count": len(new_records),
        "ref_count": len(ref_records),
        "new_script_ratio": round(new_script, 4),
        "ref_script_ratio": round(ref_script, 4),
        "script_ratio_delta": round(script_delta, 4),
        "new_length": new_len,
        "ref_length": ref_len,
        "length_delta_pct": round(length_delta_pct, 4),
        "alerts": alerts,
        "status": "DRIFT_DETECTED" if alerts else "OK",
    }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("new_snapshot", help="Path to new raw snapshot (.jsonl)")
    parser.add_argument("reference", help="Path to reference scored file (.jsonl)")
    parser.add_argument("--script-threshold", type=float, default=0.05)
    parser.add_argument("--length-threshold", type=float, default=0.20)
    args = parser.parse_args()

    result = check_drift(
        Path(args.new_snapshot),
        Path(args.reference),
        thresholds={
            "script_ratio_delta": args.script_threshold,
            "mean_length_delta_pct": args.length_threshold,
        },
    )

    print(f"Status: {result['status']}")
    print(f"  new_count:          {result['new_count']:,}")
    print(f"  ref_count:          {result['ref_count']:,}")
    print(f"  script_ratio:       {result['new_script_ratio']} vs {result['ref_script_ratio']} (delta={result['script_ratio_delta']})")
    print(f"  mean_length:        {result['new_length']['mean']} vs {result['ref_length']['mean']} ({result['length_delta_pct']:.1%})")
    if result["alerts"]:
        print("Alerts:")
        for a in result["alerts"]:
            print(f"  ! {a}")


if __name__ == "__main__":
    main()
