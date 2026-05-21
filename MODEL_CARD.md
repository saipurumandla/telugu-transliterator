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
  - wer
---

# TeluguTransliterator

Converts colloquial Romanized Telugu (Tenglish) to Telugu Unicode script — handles chat-style spellings, code-mix English passthrough, and informal Urdu loanwords common in everyday Telugu conversation.

Fine-tuned from [google/byt5-small](https://huggingface.co/google/byt5-small) on 6.1M transliteration pairs.

**[Try the live demo →](https://huggingface.co/spaces/harinpurumandla/telugu-transliterator)**

---

## What is Tenglish?

Tenglish is how most Telugu speakers actually type — using Roman letters to write Telugu words, mixed with English. It's the language of WhatsApp messages, Twitter/X posts, and YouTube comments. Standard transliteration tools built for formal input break on this style. This model is trained specifically on it.

---

## Example Outputs

| Input (Tenglish) | Output (Telugu) | Notes |
|---|---|---|
| nenu Telugu maatladutaanu | నేను తెలుగు మాట్లాడుతాను | standard sentence |
| ela unnav bro | ఎలా ఉన్నావ్ బ్రో | English loanword transliterated |
| pakka cheppindi | పక్కా చెప్పింది | Urdu loanword |
| ikkade unna | ఇక్కడే ఉన్న | chat shortening |
| nenu school ki vellanu | నేను స్కూల్ కి వెళ్ళాను | mixed formal/colloquial |
| super ga undi | సూపర్ గా ఉంది | English loanword |
| em chestunnav | ఏం చేస్తున్నావ్ | casual greeting |
| konchem wait cheyyi | కొంచెం వెయిట్ చెయ్యి | code-mix |
| anthey nuvvu cheppindi correct | అంతే నువ్వు చెప్పింది కరెక్ట్ | antey vs anthey disambiguation |
| sare raa intiki | సరే రా ఇంటికి | doubled vowel (raa→రా) |

---

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
# నేను తెలుగు మాట్లాడుతాను

print(transliterate("ela unnav bro"))
# ఎలా ఉన్నావ్ బ్రో

print(transliterate("pakka cheppindi"))
# పక్కా చెప్పింది
```

### Batch inference

```python
texts = ["nenu veltunna", "em chesav", "konchem wait cheyyi"]
results = [transliterate(t) for t in texts]
```

### GPU inference

```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def transliterate_gpu(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs, max_length=128, num_beams=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Model Details

| Property | Value |
|---|---|
| Base model | google/byt5-small |
| Architecture | Byte-level T5 (seq2seq) |
| Parameters | ~300M |
| Training data | 6,164,876 pairs across 5 sources |
| Training | 1 epoch, bf16, dual RTX 3090 (~10 hours) |
| Decoding | Greedy (num_beams=1) |
| Max input length | 128 tokens (bytes) |
| License | Apache-2.0 |

**Why ByT5?** Telugu Unicode has 128+ characters. Character-level tokenizers fragment Telugu script unpredictably. ByT5 operates at the byte level — no vocabulary mismatch, no unknown tokens for any Telugu character.

---

## Evaluation

Evaluated on a held-out test set (10,000 randomly sampled pairs), greedy decoding.

| Slice | CER ↓ | WER ↓ | n |
|-------|--------|--------|---|
| **overall** | **7.08%** | **32.14%** | 10,000 |
| formal | 3.25% | 16.90% | 71 |
| colloquial | 3.09% | 18.49% | 3,656 |
| long_sentence (>40 words) | 16.21% | 30.36% | 206 |

**CER** (Character Error Rate) is the primary metric — one wrong character in a multi-character Telugu grapheme cluster counts as one error, which is much more lenient than WER.

The high WER on long sentences is expected: ByT5's 1024-byte window truncates very long inputs. For sentences under ~40 words, CER stays under 10%.

---

## Training Data

| Source | Pairs | License |
|--------|-------|---------|
| Aksharantar | 4,709,570 | CC-BY-4.0 |
| Samanantar | 904,342 | CC-BY-4.0 |
| Wikipedia Telugu | 458,409 | CC-BY-SA-3.0 |
| Dakshina | 89,134 | CC-BY-SA-4.0 |
| Synthetic (Ollama/Gemma4) | 3,421 | internal |
| **Total** | **6,164,876** | |

Full dataset: [harinpurumandla/telugu-transliterator-dataset](https://huggingface.co/datasets/harinpurumandla/telugu-transliterator-dataset)

---

## Known Limitations

- **Long sentences (>40 words):** CER ~16% — ByT5 1024-byte window truncates inputs. Split long text into sentences before transliterating.
- **Highly informal shortenings:** Spellings not seen in training (e.g., extremely novel slang) may be passed through unchanged or mis-transliterated.
- **Dialect:** Primarily trained on standard South Indian Telugu. Regional dialect variance is limited.
- **Speed:** ByT5 generates one byte at a time — slower than character-level or subword models. Expect ~0.5–2 seconds per sentence on CPU.

---

## Intended Use

- Telugu keyboard input assistance
- Social media content processing (comments, posts in Tenglish)
- Dataset creation for downstream Telugu NLP tasks
- Research on low-resource Indic transliteration

**Not intended for:** Medical, legal, or safety-critical text where errors could cause harm.

---

## Citation

```bibtex
@misc{telugu-transliterator-2026,
  title={TeluguTransliterator: Local Transliteration for Colloquial Romanized Telugu},
  author={Purumandla, Sai Harin Kumar Reddy},
  year={2026},
  url={https://huggingface.co/harinpurumandla/telugu-transliterator}
}
```
