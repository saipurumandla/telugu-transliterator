"""
Compare v4 ByT5 model vs Gemma4 (Ollama) on transliteration quality.

Usage:
    python -m eval.compare_gemma4
    python -m eval.compare_gemma4 --samples 200 --model-dir train/checkpoints_v4/best
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_GEMMA_MODEL = "gemma4:31b"

SYSTEM_MSG = (
    "You are a Telugu transliteration engine. "
    "Convert Tenglish (Romanized Telugu) to Telugu Unicode script. "
    "Reply with ONLY the Telugu Unicode text — nothing else."
)


# ── CER ────────────────────────────────────────────────────────────────────────

def _cer(pred: str, ref: str) -> float:
    ref_chars, pred_chars = list(ref), list(pred)
    m, n = len(ref_chars), len(pred_chars)
    if m == 0:
        return 0.0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if ref_chars[i - 1] == pred_chars[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / m


def compute_cer(preds: list[str], refs: list[str]) -> float:
    total_chars = total_errors = 0
    for pred, ref in zip(preds, refs):
        total_chars += len(ref)
        total_errors += _cer(pred, ref) * len(ref)
    return total_errors / max(total_chars, 1)


# ── Sample test set ────────────────────────────────────────────────────────────

def _is_clean(r: dict) -> bool:
    roman = r.get("roman_text", "")
    if not (8 <= len(roman) <= 60):
        return False
    # no digits, no special punctuation that confuses LLMs
    if any(c in roman for c in "0123456789@#$%^&*()[]{}|\\<>"):
        return False
    # must be mostly ASCII letters
    alpha = sum(1 for c in roman if c.isalpha())
    if alpha / max(len(roman), 1) < 0.7:
        return False
    return True


def sample_test(test_file: str, n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    with open(test_file, encoding="utf-8") as fh:
        records = [json.loads(l) for l in fh]
    clean = [r for r in records if _is_clean(r)]
    return rng.sample(clean, min(n, len(clean)))


# ── ByT5 inference ─────────────────────────────────────────────────────────────

def run_byt5(
    samples: list[dict],
    model_dir: str,
    batch_size: int = 32,
) -> list[str]:
    print(f"Loading ByT5 from {model_dir} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    model.to(device).eval()

    preds = []
    texts = [s["roman_text"] for s in samples]
    print(f"Running ByT5 on {len(texts)} examples (batch={batch_size}) ...")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_length=128, num_beams=4)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        print(f"  ByT5: {min(i + batch_size, len(texts))}/{len(texts)}")

    del model
    torch.cuda.empty_cache()
    return preds


# ── Gemma4 inference ───────────────────────────────────────────────────────────

def _gemma4_predict(roman: str, ollama_chat_url: str, gemma_model: str, timeout: int = 90) -> str | None:
    try:
        resp = requests.post(
            ollama_chat_url,
            json={
                "model": gemma_model,
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": roman},
                ],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json().get("message", {}).get("content", "").strip()
        return text.split("\n")[0].strip()
    except Exception as e:
        print(f"    Gemma4 error: {e}")
        return None


def run_gemma4(samples: list[dict], ollama_host: str, gemma_model: str) -> list[str | None]:
    chat_url = f"{ollama_host.rstrip('/')}/api/chat"
    print(f"Running {gemma_model} on {len(samples)} examples (sequential) ...")
    preds = []
    for i, s in enumerate(samples):
        t0 = time.time()
        pred = _gemma4_predict(s["roman_text"], chat_url, gemma_model)
        elapsed = time.time() - t0
        preds.append(pred)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Gemma4: {i + 1}/{len(samples)}  last={elapsed:.1f}s  pred={str(pred)[:40]}")
    return preds


# ── Report ─────────────────────────────────────────────────────────────────────

def report(
    samples: list[dict],
    byt5_preds: list[str],
    gemma4_preds: list[str | None],
    out_file: str | None,
) -> None:
    refs = [s["telugu_text"] for s in samples]
    valid = [(i, p) for i, p in enumerate(gemma4_preds) if p is not None]
    g4_idx = [i for i, _ in valid]
    g4_preds_clean = [p for _, p in valid]

    byt5_cer = compute_cer(byt5_preds, refs)
    g4_cer = compute_cer(g4_preds_clean, [refs[i] for i in g4_idx]) if valid else float("nan")

    print()
    print("=" * 60)
    print(f"  Model                 CER       Samples")
    print(f"  ByT5-small v4         {byt5_cer:.2%}    {len(samples)}")
    print(f"  Gemma4 31B (Ollama)   {g4_cer:.2%}    {len(valid)}")
    print("=" * 60)

    # Show 10 side-by-side examples where they differ most
    diffs = []
    for i, (s, bp, gp) in enumerate(zip(samples, byt5_preds, gemma4_preds)):
        if gp is None:
            continue
        b_err = _cer(bp, s["telugu_text"])
        g_err = _cer(gp, s["telugu_text"])
        diffs.append((abs(b_err - g_err), i, s, bp, gp))
    diffs.sort(reverse=True)

    print("\nTop 10 most divergent examples:")
    print("-" * 60)
    for _, i, s, bp, gp in diffs[:10]:
        b_err = _cer(bp, s["telugu_text"])
        g_err = _cer(gp, s["telugu_text"])
        winner = "ByT5" if b_err < g_err else "Gemma4"
        print(f"Input:  {s['roman_text'][:60]}")
        print(f"Ref:    {s['telugu_text'][:60]}")
        print(f"ByT5:   {bp[:60]}  (CER {b_err:.2%})")
        print(f"Gemma4: {gp[:60]}  (CER {g_err:.2%})  <- {winner} wins")
        print()

    results = {
        "byt5_cer": round(byt5_cer, 4),
        "gemma4_cer": round(g4_cer, 4),
        "samples": len(samples),
        "gemma4_valid": len(valid),
        "examples": [
            {
                "roman": s["roman_text"],
                "ref": s["telugu_text"],
                "byt5": bp,
                "gemma4": gp,
                "byt5_cer": round(_cer(bp, s["telugu_text"]), 4),
                "gemma4_cer": round(_cer(gp, s["telugu_text"]), 4) if gp else None,
            }
            for s, bp, gp in zip(samples, byt5_preds, gemma4_preds)
        ],
    }

    if out_file:
        Path(out_file).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results saved to {out_file}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--model-dir", default="train/checkpoints_v4/best")
    parser.add_argument("--test-file", default="data/processed_v3/test.jsonl")
    parser.add_argument("--out", default="eval/compare_gemma4_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ollama-host", required=True, help="Ollama base URL, e.g. http://localhost:11434")
    parser.add_argument("--gemma-model", default=DEFAULT_GEMMA_MODEL)
    args = parser.parse_args()

    print(f"Sampling {args.samples} examples from {args.test_file} ...")
    samples = sample_test(args.test_file, args.samples, args.seed)

    byt5_preds = run_byt5(samples, args.model_dir)
    gemma4_preds = run_gemma4(samples, args.ollama_host, args.gemma_model)

    report(samples, byt5_preds, gemma4_preds, args.out)


if __name__ == "__main__":
    main()
