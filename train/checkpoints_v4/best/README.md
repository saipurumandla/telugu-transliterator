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

# Telugu Transliterator v1.1 (Tenglish → Telugu)

Converts colloquial Romanized Telugu (Tenglish) to Telugu Unicode script.

Fine-tuned from [google/byt5-small](https://huggingface.co/google/byt5-small) on 5.6M transliteration pairs including 6,364 targeted synthetic pairs generated to fix colloquial error patterns identified in v1.0.

---

## Example Outputs

| Input (Tenglish) | Output (Telugu) | Notes |
|---|---|---|
| nenu telugu matladatanu | నేను తెలుగు మాట్లాడతాను | standard sentence |
| ela unnav bro | ఎలా ఉన్నావ్ బ్రో | casual greeting with English code-mix |
| chala bagundi aa movie | చాలా బాగుంది ఆ మూవీ | informal review |
| repu veltunnanu | రేపు వెళ్తున్నాను | near-future plan |
| antey em cheppali | అంటే ఏం చెప్పాలి | disambiguation: antey = "means" |
| ante idi chalu | అంతే ఇది చాలు | disambiguation: ante = "that's all" |
| naaku teliyadu | నాకు తెలియదు | dative suffix: naaku |
| pakka correct ga cheppadu | పక్కా కరెక్ట్ గా చెప్పాడు | Urdu loanword: pakka |

---

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("harinpurumandla/telugu-transliterator")
model = AutoModelForSeq2SeqLM.from_pretrained("harinpurumandla/telugu-transliterator")

def to_telugu(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(to_telugu("nenu telugu matladatanu"))
# నేను తెలుగు మాట్లాడతాను

print(to_telugu("ela unnav bro"))
# ఎలా ఉన్నావ్ బ్రో
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | google/byt5-small |
| Architecture | Byte-level T5 (seq2seq) |
| Parameters | ~300M |
| Direction | Tenglish (Roman) → Telugu script |
| Training data | 5,586,373 pairs (5 corpus sources + 6,364 synthetic) |
| Training | 1 epoch continued fine-tune, LR=1e-5, bf16, dual RTX 3090 |
| Eval CER | 1.89% |
| Decoding | Beam search (num_beams=4) |
| Max input length | 128 tokens (bytes) |
| License | Apache-2.0 |

---

## Evaluation

Evaluated on a held-out val set (5,000 sampled pairs), beam search decoding (num_beams=4).

### Version History

| Version | CER | Notes |
|---|---|---|
| v1.0 | 7.08% | Baseline — trained from scratch on 5.5M pairs |
| **v1.1** | **1.89%** | Continued fine-tune with targeted synthetic colloquial data |

### Comparison with Gemma4 31B

Evaluated on 50 clean Tenglish inputs (8–60 chars, alphabetic):

| Model | CER | Parameters |
|---|---|---|
| **ByT5-small v1.1 (this model)** | **2.70%** | 300M |
| Gemma4 31B (zero-shot, Ollama) | 8.48% | 31B |

ByT5-small achieves **3× lower error rate at 1% of the parameter count**. The key difference: Gemma4 normalizes colloquial spellings toward standard Telugu (e.g., `chupiyaru` → చూపించారు), while this model faithfully preserves the informal romanization convention (→ చుపియారు). For Tenglish transliteration, faithfulness to the input spelling is the correct behavior.

### Per-Slice (v1.0 baseline for reference)

| Slice | v1.0 CER | Coverage |
|---|---|---|
| Overall | 7.08% | 10,000 pairs |
| Formal | 3.25% | 71 pairs |
| Colloquial | 3.09% | 3,656 pairs |
| Long sentence | 16.21% | 206 pairs |

---

## Synthetic Data

v1.1 added 6,364 targeted synthetic pairs generated with local Ollama gemma4:31b covering 9 confusion categories identified from v1.0 error analysis:

| Category | Pairs | Focus |
|---|---|---|
| antey_disambiguation | 704 | అంటే (means) vs అంతే (that's all) in context |
| dative_suffix_context | 701 | naaku / naku / ki suffix forms |
| colloquial_chat | 705 | WhatsApp-style greetings, questions, reactions |
| urdu_loanwords | 1,000 | pakka, bilkul, mast, zabardast in Telugu sentences |
| code_mix | 1,000 | English words with Telugu morphological markers |
| long_sentences | 800 | Full sentences >15 words |
| reactions_expressions | 701 | Emotional interjections and expressive phrases |
| continuous_negation | 403 | Negation + continuous tense combinations |
| long_vowel_disambiguation | 350 | Phonemically distinct long vs short vowels |

All synthetic pairs were validated for Telugu script density (≥10%), length ratio, and ASCII content before inclusion.

---

## Known Limitations

- **Long sentences (>40 words):** ByT5 1024-byte window truncates inputs. Split long text before transliterating.
- **Rare proper nouns:** Unusual place names and personal names may be inconsistently transliterated.
- **Spelling convention:** When multiple valid Tenglish spellings exist for a word, the model outputs the most common convention learned from training data.
- **Pure English words:** Code-mixed English words pass through but capitalization may not be preserved.

---

## Companion Model

For the reverse direction (Telugu script → Tenglish), see [harinpurumandla/telugu-to-tenglish](https://huggingface.co/harinpurumandla/telugu-to-tenglish).

---

## Citation

```bibtex
@misc{telugu-transliterator-2026,
  title={Telugu Transliterator: Tenglish to Telugu Unicode Transliteration},
  author={Purumandla, Sai Harin Kumar Reddy},
  year={2026},
  url={https://huggingface.co/harinpurumandla/telugu-transliterator}
}
```
