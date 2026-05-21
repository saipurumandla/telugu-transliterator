"""CLI for local offline inference — Tenglish to Telugu transliteration."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_CHECKPOINT = "train/checkpoints_v3/checkpoint-80000"


def load_model(checkpoint_dir: str):
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        print(f"Checkpoint not found: {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to(device)
    model.eval()
    return tokenizer, model, device


def transliterate(text: str, tokenizer, model, device: str, max_length: int = 128) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tenglish → Telugu transliteration")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("text", nargs="*", help="Text to transliterate (or stdin if omitted)")
    args = parser.parse_args()

    print("Loading model ...", file=sys.stderr)
    tokenizer, model, device = load_model(args.checkpoint)
    print(f"Ready on {device}.", file=sys.stderr)

    if args.text:
        inputs = [" ".join(args.text)]
    else:
        inputs = [line.strip() for line in sys.stdin if line.strip()]

    for text in inputs:
        result = transliterate(text, tokenizer, model, device, args.max_length)
        print(result)


if __name__ == "__main__":
    main()
