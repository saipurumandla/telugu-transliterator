"""Generate MODEL_CARD.md from eval results and training config."""

import json
import yaml
from pathlib import Path


TEMPLATE = """\
---
language: te
tags:
  - transliteration
  - telugu
  - tenglish
  - byt5
license: apache-2.0
---

# TeluguTransliterator

Converts colloquial Romanized Telugu (Tenglish) to Telugu Unicode script.
Fine-tuned from [google/byt5-small](https://huggingface.co/google/byt5-small).

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("harinpurumandla/telugu-transliterator")
model = AutoModelForSeq2SeqLM.from_pretrained("harinpurumandla/telugu-transliterator")

def transliterate(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(transliterate("nenu Telugu maatladutaanu"))
print(transliterate("ela unnav bro"))
print(transliterate("pakka cheppindi"))
```

## Model Details

- **Base model:** google/byt5-small (byte-level T5, no tokenizer issues for Telugu)
- **Training data:** {total_pairs:,} pairs across {num_sources} sources
- **Training:** 1 epoch, bf16, dual RTX 3090
- **License:** Apache-2.0

## Evaluation

| Slice | CER | WER | n |
|-------|-----|-----|---|
| overall | {cer_overall} | {wer_overall} | {n_overall:,} |
| formal | {cer_formal} | {wer_formal} | {n_formal:,} |
| colloquial | {cer_colloquial} | {wer_colloquial} | {n_colloquial:,} |
| long_sentence | {cer_long} | {wer_long} | {n_long:,} |

Evaluated with greedy decoding (num_beams=1), max_length=128.

## Known Limitations

- Long sentences (>40 words): CER ~16% — ByT5 byte limit truncates very long inputs
- Very informal shortenings not in training data may be passed through unchanged
- Primarily trained on South Indian Telugu dialect patterns

## Data Sources

- Aksharantar (CC-BY-4.0)
- Samanantar (CC-BY-4.0)
- Wikipedia Telugu (CC-BY-SA-3.0)
- Dakshina (CC-BY-SA-4.0)
- Synthetic pairs generated via local Ollama (internal)

## Citation

```
@misc{{telugu-transliterator-2026,
  title={{TeluguTransliterator: Local Transliteration for Colloquial Romanized Telugu}},
  author={{Purumandla, Sai}},
  year={{2026}},
  url={{https://huggingface.co/harinpurumandla/telugu-transliterator}}
}}
```
"""


def make_model_card(
    eval_file: str = "eval/v3_eval.json",
    config_file: str = "train/config.yaml",
    output: str = "MODEL_CARD.md",
) -> None:
    eval_path = Path(eval_file)
    if eval_path.exists():
        results = json.loads(eval_path.read_text(encoding="utf-8"))
    else:
        results = {}

    def fmt_cer(key: str) -> str:
        v = results.get(key, {}).get("cer")
        return f"{v*100:.2f}%" if v is not None else "—"

    def fmt_wer(key: str) -> str:
        v = results.get(key, {}).get("wer")
        return f"{v*100:.2f}%" if v is not None else "—"

    def n(key: str) -> int:
        return results.get(key, {}).get("count", 0)

    card = TEMPLATE.format(
        total_pairs=6_164_876,
        num_sources=5,
        cer_overall=fmt_cer("overall"),
        wer_overall=fmt_wer("overall"),
        n_overall=n("overall"),
        cer_formal=fmt_cer("formal"),
        wer_formal=fmt_wer("formal"),
        n_formal=n("formal"),
        cer_colloquial=fmt_cer("colloquial"),
        wer_colloquial=fmt_wer("colloquial"),
        n_colloquial=n("colloquial"),
        cer_long=fmt_cer("long_sentence"),
        wer_long=fmt_wer("long_sentence"),
        n_long=n("long_sentence"),
    )

    Path(output).write_text(card, encoding="utf-8")
    print(f"Model card written to {output}")


if __name__ == "__main__":
    make_model_card()
