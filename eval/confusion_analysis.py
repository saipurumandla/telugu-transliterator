"""
Confusion set analysis — finds systematic error patterns in model predictions.
Samples from the test set, runs inference, categorises failures by type.
Output goes to reports/benchmarks/confusion_report.json.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TELUGU_START = 0x0C00
TELUGU_END = 0x0C7F

SAMPLE_SIZE = 2000
FAIL_CER_THRESHOLD = 0.20
BATCH_SIZE = 64
SEED = 42


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    m, n = len(ref), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if ref[i-1] == pred[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n] / m


def _length_bucket(text: str) -> str:
    n = len(text)
    if n < 10:
        return "short (<10)"
    if n < 30:
        return "medium (10-30)"
    if n < 80:
        return "long (30-80)"
    return "very_long (80+)"


def _is_mostly_telugu(text: str) -> bool:
    if not text:
        return False
    te = sum(1 for c in text if TELUGU_START <= ord(c) <= TELUGU_END)
    return te / len(text) >= 0.3


def _char_substitution_errors(pred: str, ref: str) -> list[tuple[str, str]]:
    """Extract common single-char substitution patterns from two strings."""
    errors = []
    min_len = min(len(pred), len(ref))
    for i in range(min_len):
        if pred[i] != ref[i]:
            errors.append((ref[i], pred[i]))
    return errors


def run_analysis(
    model_dir: str,
    test_file: str,
    output_path: str,
    sample_size: int = SAMPLE_SIZE,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    print(f"Loading test pairs from {test_file} ...")
    all_pairs = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_pairs.append(json.loads(line))

    random.seed(SEED)
    sample = random.sample(all_pairs, min(sample_size, len(all_pairs)))
    print(f"  Analysing {len(sample):,} sampled pairs ...")

    def predict_batch(romans: list[str]) -> list[str]:
        inputs = tokenizer(
            romans, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128, num_beams=1)
        preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        return preds

    results = []
    for i in range(0, len(sample), BATCH_SIZE):
        batch = sample[i:i + BATCH_SIZE]
        romans = [p.get("roman_text", "") for p in batch]
        preds = predict_batch(romans)
        for pair, pred in zip(batch, preds):
            ref = pair.get("telugu_text", "")
            c = cer(pred, ref)
            results.append({
                "roman": pair.get("roman_text", ""),
                "ref": ref,
                "pred": pred,
                "cer": round(c, 4),
                "source": pair.get("source_name", ""),
                "pair_source": pair.get("pair_source", ""),
            })
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  {min(i + BATCH_SIZE, len(sample))}/{len(sample)}")

    failing = [r for r in results if r["cer"] >= FAIL_CER_THRESHOLD]
    print(f"\nFailing pairs (CER >= {FAIL_CER_THRESHOLD}): {len(failing)} of {len(results)}")

    # --- Categorise failures ---

    by_source: dict[str, list] = defaultdict(list)
    by_length: dict[str, list] = defaultdict(list)
    by_pair_source: dict[str, list] = defaultdict(list)
    char_errors: Counter = Counter()

    for r in failing:
        by_source[r["source"]].append(r["cer"])
        by_length[_length_bucket(r["roman"])].append(r["cer"])
        by_pair_source[r["pair_source"]].append(r["cer"])
        for ref_c, pred_c in _char_substitution_errors(r["pred"], r["ref"]):
            if _is_mostly_telugu(ref_c) or _is_mostly_telugu(pred_c):
                char_errors[f"{ref_c}→{pred_c}"] += 1

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    report = {
        "sample_size": len(results),
        "fail_threshold": FAIL_CER_THRESHOLD,
        "failing_count": len(failing),
        "failing_rate": round(len(failing) / len(results), 4),
        "overall_cer": round(sum(r["cer"] for r in results) / len(results), 4),
        "by_source": {k: {"count": len(v), "avg_cer": avg(v)} for k, v in sorted(by_source.items())},
        "by_length": {k: {"count": len(v), "avg_cer": avg(v)} for k, v in sorted(by_length.items())},
        "by_pair_source": {k: {"count": len(v), "avg_cer": avg(v)} for k, v in sorted(by_pair_source.items())},
        "top_char_errors": [{"pattern": p, "count": c} for p, c in char_errors.most_common(20)],
        "worst_examples": sorted(failing, key=lambda r: r["cer"], reverse=True)[:20],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n=== Confusion Report ===")
    print(f"Overall CER on sample: {report['overall_cer']}")
    print(f"Failing rate: {report['failing_rate']:.1%}")
    print(f"\nBy source:")
    for src, m in report["by_source"].items():
        print(f"  {src:<20} count={m['count']:>5}  avg_cer={m['avg_cer']}")
    print(f"\nBy input length:")
    for length, m in report["by_length"].items():
        print(f"  {length:<25} count={m['count']:>5}  avg_cer={m['avg_cer']}")
    print(f"\nTop character errors:")
    import sys as _sys
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for e in report["top_char_errors"][:10]:
        print(f"  {e['pattern']}  x{e['count']}")
    print(f"\nSaved to {output_path}")

    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="train/checkpoints/best")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--output", default="reports/benchmarks/confusion_report.json")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    args = parser.parse_args()
    run_analysis(args.model_dir, args.test_file, args.output, args.sample_size)


if __name__ == "__main__":
    main()
