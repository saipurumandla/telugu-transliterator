"""Compare two eval result JSON files and print a delta report."""

import json
import sys
from pathlib import Path


def compare(baseline_path: str, v2_path: str) -> None:
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(v2_path, encoding="utf-8") as f:
        v2 = json.load(f)

    all_slices = sorted(set(list(baseline.keys()) + list(v2.keys())))

    print(f"{'Slice':<20} {'Baseline CER':>12} {'v2 CER':>10} {'Delta':>10} {'Status':>8}")
    print("-" * 65)

    for slice_name in all_slices:
        b = baseline.get(slice_name, {})
        v = v2.get(slice_name, {})
        b_cer = b.get("cer")
        v_cer = v.get("cer")

        if b_cer is None or v_cer is None:
            print(f"  {slice_name:<18} {'N/A':>12} {'N/A':>10}")
            continue

        delta = v_cer - b_cer
        pct = (delta / b_cer * 100) if b_cer > 0 else 0
        status = "BETTER" if delta < -0.002 else ("WORSE" if delta > 0.002 else "same")
        print(f"  {slice_name:<18} {b_cer:>12.4f} {v_cer:>10.4f} {delta:>+10.4f} ({pct:>+.1f}%)  {status}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="reports/benchmarks/baseline_eval.json")
    parser.add_argument("--v2", default="reports/benchmarks/v2_eval.json")
    args = parser.parse_args()

    if not Path(args.v2).exists():
        print(f"v2 results not found at {args.v2}")
        print("Run eval/evaluate.py with --output reports/benchmarks/v2_eval.json first.")
        sys.exit(1)

    print(f"\nBaseline: {args.baseline}")
    print(f"v2:       {args.v2}\n")
    compare(args.baseline, args.v2)


if __name__ == "__main__":
    main()
