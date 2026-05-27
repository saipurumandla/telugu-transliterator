---
language: te
tags:
  - transliteration
  - telugu
  - tenglish
  - byt5
  - romanized-telugu
  - indicnlp
  - telugu-unicode
  - low-resource
  - indic-languages
  - seq2seq
license: apache-2.0
datasets:
  - harinpurumandla/telugu-transliterator-dataset
metrics:
  - cer
---

# Telugu to Tenglish Transliterator v1.1

Converts Telugu Unicode script to colloquial Romanized Telugu (Tenglish) — the reverse direction of [harinpurumandla/telugu-transliterator](https://huggingface.co/harinpurumandla/telugu-transliterator).

Fine-tuned from [google/byt5-small](https://huggingface.co/google/byt5-small) on 5.6M transliteration pairs including 6,364 targeted synthetic pairs covering colloquial confusion patterns.

---

## Example Outputs

| Input (Telugu) | Output (Tenglish) | Notes |
|---|---|---|
| నేను తెలుగు మాట్లాడతాను | nenu telugu matladatanu | standard sentence |
| ఏమి చేస్తున్నావు | emi chestunavu | casual question |
| నువ్వు ఎలా ఉన్నావు | nuvvu ela unnavu | greeting |
| చాలా బాగుంది | chala bagundi | common phrase |
| రేపు కలుద్దాం | repu kaluddam | informal plan |
| సరే అలాగే చేస్తాను | sare alage chestanu | agreement |
| పక్కా చెప్తున్నా | pakka cheptunna | Urdu loanword |

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("harinpurumandla/telugu-to-tenglish")
model = AutoModelForSeq2SeqLM.from_pretrained("harinpurumandla/telugu-to-tenglish")

def to_tenglish(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(to_tenglish("నేను తెలుగు మాట్లాడతాను"))
# nenu telugu matladatanu

print(to_tenglish("చాలా బాగుంది"))
# chala bagundi
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | google/byt5-small |
| Architecture | Byte-level T5 (seq2seq) |
| Parameters | ~300M |
| Direction | Telugu script → Tenglish (Roman) |
| Training data | 5,586,373 pairs (5 corpus sources + 6,364 synthetic) |
| Training | 1 epoch continued fine-tune, LR=1e-5, bf16, dual RTX 3090 |
| Eval CER | 16.69% |
| Decoding | Beam search (num_beams=4) |
| Max input length | 128 tokens (bytes) |
| License | Apache-2.0 |

---

## Evaluation

Evaluated on a held-out val set (5,000 sampled pairs), beam search decoding (num_beams=4).

### Version History

| Version | CER | Notes |
|---|---|---|
| v1.0 | 16.53% | Baseline — trained from scratch on 5.5M pairs |
| **v1.1** | **16.69%** | Continued fine-tune with targeted synthetic colloquial data |

The CER is essentially flat between versions — the synthetic colloquial data did not significantly change reverse model performance. This is expected: the reverse direction (Telugu → Tenglish) has inherent ambiguity that cannot be resolved by more training data alone. The same Telugu word has multiple valid Tenglish spellings (e.g., అంతే → `anthe`, `anthey`, `antey`, `ante`), so CER against a single reference underestimates true model quality.

### Why Reverse CER is Higher than Forward

The forward model (Tenglish → Telugu) achieves 1.89% CER because Telugu Unicode is phonetically consistent — each romanization maps to one correct script form. The reverse direction has no such constraint: Tenglish spelling is not standardized, so the model must guess which of several valid conventions the reference used. A prediction of `nenu` vs `nennu` is linguistically correct but scores as an error.

---

## Known Limitations

- **Spelling convention:** Multiple valid Tenglish spellings exist for most Telugu words — the model outputs one learned convention which may differ from a user's preferred spelling.
- **Vowel length:** Long vowels (ఆ, ఈ, ఊ) may be rendered as short in output (`aa→a`, `uu→u`).
- **Suffix truncation:** Final short syllables occasionally dropped on longer words.
- **Long sentences (>40 words):** ByT5 1024-byte window truncates inputs. Split long text before transliterating.

---

## Companion Model

For the forward direction (Tenglish → Telugu script), see [harinpurumandla/telugu-transliterator](https://huggingface.co/harinpurumandla/telugu-transliterator).

---

## Citation

```bibtex
@misc{telugu-to-tenglish-2026,
  title={Telugu to Tenglish: Reverse Transliteration for Telugu Unicode Script},
  author={Purumandla, Sai Harin Kumar Reddy},
  year={2026},
  url={https://huggingface.co/harinpurumandla/telugu-to-tenglish}
}
```
