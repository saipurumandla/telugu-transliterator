import json
import os
import sys
from pathlib import Path

import torch
import yaml
from jiwer import wer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


TELUGU_START = 0x0C00
TELUGU_END = 0x0C7F


def _has_english_words(text: str, min_ratio: float = 0.2) -> bool:
    words = text.split()
    if not words:
        return False
    english = sum(1 for w in words if all(c.isascii() and c.isalpha() for c in w) and len(w) > 1)
    return english / len(words) >= min_ratio


def _has_actual_english(text: str, english_words: list[str]) -> bool:
    words = [w.lower().strip(".,!?") for w in text.split()]
    return any(w in english_words for w in words)


def load_test_pairs(test_file: str) -> list[dict]:
    pairs = []
    with open(test_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def filter_slice(pairs: list[dict], slice_cfg: dict) -> list[dict]:
    source_names = slice_cfg.get("source_names")
    pair_sources = slice_cfg.get("pair_sources")
    filter_type = slice_cfg.get("filter")
    min_roman_len = slice_cfg.get("min_roman_length", 0)
    min_english_ratio = slice_cfg.get("min_english_word_ratio", 0.2)

    result = []
    for pair in pairs:
        if source_names and pair.get("source_name") not in source_names:
            continue
        if pair_sources and pair.get("pair_source") not in pair_sources:
            continue
        roman = pair.get("roman_text") or ""
        if len(roman) < min_roman_len:
            continue
        if filter_type == "roman_has_english_words":
            if not _has_english_words(roman, min_english_ratio):
                continue
        if filter_type == "roman_has_actual_english":
            english_words = slice_cfg.get("english_words", [])
            if not _has_actual_english(roman, english_words):
                continue
        result.append(pair)
    return result


def cer(pred: str, ref: str) -> float:
    ref_chars = list(ref)
    pred_chars = list(pred)
    m, n = len(ref_chars), len(pred_chars)
    if m == 0:
        return 0.0 if n == 0 else 1.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref_chars[i - 1] == pred_chars[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / m


def evaluate_model(
    model_dir: str,
    test_file: str,
    slices_file: str,
    batch_size: int = 64,
    max_length: int = 128,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_dir} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    print(f"Loading test pairs from {test_file} ...")
    all_pairs = load_test_pairs(test_file)
    print(f"  {len(all_pairs):,} test pairs")

    with open(slices_file, encoding="utf-8") as f:
        slices_cfg = yaml.safe_load(f)

    def predict_batch(roman_texts: list[str]) -> list[str]:
        inputs = tokenizer(
            roman_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=1,
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def eval_pairs(pairs: list[dict], label: str) -> dict:
        if not pairs:
            return {"count": 0, "cer": None, "wer": None}

        romans = [p.get("roman_text", "") for p in pairs]
        refs = [p.get("telugu_text", "") for p in pairs]
        preds = []

        for i in range(0, len(romans), batch_size):
            batch = romans[i:i + batch_size]
            preds.extend(predict_batch(batch))
            if (i // batch_size) % 10 == 0:
                print(f"  {label}: {min(i + batch_size, len(romans))}/{len(romans)}")

        total_cer = sum(cer(p, r) for p, r in zip(preds, refs)) / len(pairs)
        word_error = wer([r for r in refs], [p for p in preds])
        exact = sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip()) / len(pairs)

        return {
            "count": len(pairs),
            "cer": round(total_cer, 4),
            "wer": round(word_error, 4),
            "exact_match": round(exact, 4),
        }

    results: dict[str, dict] = {}

    # Overall
    print("Evaluating full test set ...")
    results["overall"] = eval_pairs(all_pairs, "overall")

    # Per slice
    for slice_cfg in slices_cfg.get("slices", []):
        name = slice_cfg["name"]
        subset = filter_slice(all_pairs, slice_cfg)
        print(f"Evaluating slice '{name}' ({len(subset)} pairs) ...")
        results[name] = eval_pairs(subset, name)

    return results


def _find_model_dir(requested: str) -> str:
    p = Path(requested)
    if p.exists():
        return str(p)
    # Fall back to latest checkpoint directory
    ckpt_root = Path("train/checkpoints")
    checkpoints = sorted(ckpt_root.glob("checkpoint-*"),
                         key=lambda x: int(x.name.split("-")[1]))
    if checkpoints:
        fallback = str(checkpoints[-1])
        print(f"Model dir '{requested}' not found — using {fallback}")
        return fallback
    raise FileNotFoundError(f"No model found at '{requested}' and no checkpoints in {ckpt_root}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="train/checkpoints/best")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--slices-file", default="eval/eval_slices.yaml")
    parser.add_argument("--output", default="reports/benchmarks/eval_results.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use")
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    results = evaluate_model(
        _find_model_dir(args.model_dir),
        args.test_file,
        args.slices_file,
        args.batch_size,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Results ===")
    for slice_name, metrics in results.items():
        print(f"  {slice_name:<20} CER={metrics.get('cer', 'N/A')}  WER={metrics.get('wer', 'N/A')}  n={metrics.get('count', 0)}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
