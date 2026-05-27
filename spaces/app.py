"""Gradio demo for Telugu transliteration — both directions."""

import re as _re

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

FORWARD_MODEL_ID = "harinpurumandla/telugu-transliterator"
REVERSE_MODEL_ID = "harinpurumandla/telugu-to-tenglish"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading models on {device} ...")

fwd_tokenizer = AutoTokenizer.from_pretrained(FORWARD_MODEL_ID)
fwd_model = AutoModelForSeq2SeqLM.from_pretrained(FORWARD_MODEL_ID).to(device).eval()

rev_tokenizer = AutoTokenizer.from_pretrained(REVERSE_MODEL_ID)
rev_model = AutoModelForSeq2SeqLM.from_pretrained(REVERSE_MODEL_ID).to(device).eval()

print("Models ready.")

# ── Post-processing ────────────────────────────────────────────────────────────

_MATRA_ARTIFACT = [
    ("ాఆ", "ా"), ("ిఇ", "ి"), ("ీఈ", "ీ"), ("ుఉ", "ు"), ("ూఊ", "ూ"),
    ("ెఎ", "ె"), ("ేఏ", "ే"), ("ొఒ", "ొ"), ("ోఓ", "ో"),
]
_EY_SUFFIX = _re.compile(r'(?<=[a-z])ey\b', _re.IGNORECASE)


def _fix_artifacts(text: str) -> str:
    for bad, good in _MATRA_ARTIFACT:
        text = text.replace(bad, good)
    return text


def _normalize_tenglish(text: str) -> str:
    return _EY_SUFFIX.sub('e', text)


# ── Inference ──────────────────────────────────────────────────────────────────

def _run(text: str, tokenizer, model) -> str:
    text = text.strip()
    if not text:
        return ""
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def to_telugu(text: str) -> str:
    text = _normalize_tenglish(text)
    return _fix_artifacts(_run(text, fwd_tokenizer, fwd_model))


def to_tenglish(text: str) -> str:
    return _run(text, rev_tokenizer, rev_model)


# ── Content ────────────────────────────────────────────────────────────────────

FWD_EXAMPLES = [
    ["nenu Telugu maatladutaanu"],
    ["ela unnav bro"],
    ["pakka cheppindi"],
    ["super ga undi"],
    ["em chestunnav"],
    ["konchem wait cheyyi"],
    ["anthey nuvvu cheppindi correct"],
    ["naaku teliyadu"],
    ["repu veltunna"],
    ["bilkul correct ga cheppadu"],
]

REV_EXAMPLES = [
    ["నేను తెలుగు మాట్లాడతాను"],
    ["ఎలా ఉన్నావ్"],
    ["చాలా బాగుంది"],
    ["ఏం చేస్తున్నావు"],
    ["రేపు కలుద్దాం"],
    ["నాకు తెలియదు"],
    ["సరే అలాగే చేస్తాను"],
    ["ఇక్కడికి రా"],
    ["కొంచెం ఆగు"],
    ["పక్కా చెప్తున్నా"],
]

REFERENCE = """
### Vowels

| Tenglish | Telugu | Sound |
|---|---|---|
| a | అ | short a |
| aa | ఆ | long aa |
| i | ఇ | short i |
| ee / ii | ఈ | long ee |
| u | ఉ | short u |
| uu / oo | ఊ | long oo |
| e | ఎ | short e |
| ee / ae | ఏ | long ae |
| ai | ఐ | ai |
| o | ఒ | short o |
| oh / O | ఓ | long oh |
| au / ow | ఔ | au |

### Key Consonant Pairs

| Tenglish | Telugu | Example |
|---|---|---|
| t (retroflex) | ట | antey అంటే |
| th (dental) | త | anthey అంతే |
| d (retroflex) | డ | adda అడ్డ |
| dh (dental) | ద | dhaari దారి |
| n | న | nenu నేను |
| sh / S | శ | shanti శాంతి |
| v / w | వ | veyyi / weyyi వెయ్యి |

### Common Words

| Tenglish | Telugu | Meaning |
|---|---|---|
| antey | అంటే | meaning / like |
| anthey | అంతే | that's all |
| ledu | లేదు | no / not there |
| undi | ఉంది | is / there is |
| ela | ఎలా | how |
| em | ఏం | what |
| pakka | పక్కా | definitely |
| konchem | కొంచెం | a little |

**Tips:**
- English words (bro, ok, super, wait) pass through unchanged
- Doubled vowels = length: `aa` = ఆ, `ee` = ఈ
- Long sentences (>40 words) may lose accuracy — split first
"""

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Telugu Transliterator", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# Telugu Transliterator

Bidirectional transliteration between Tenglish (Romanized Telugu) and Telugu Unicode script.
Handles chat-style spellings, code-mix English, and colloquial Urdu loanwords.
        """
    )

    with gr.Tabs():

        with gr.Tab("Tenglish → Telugu"):
            gr.Markdown("Type colloquial Romanized Telugu and get Telugu Unicode script.")
            with gr.Row():
                fwd_in = gr.Textbox(
                    label="Tenglish",
                    placeholder="nenu Telugu maatladutaanu",
                    lines=4,
                )
                fwd_out = gr.Textbox(label="Telugu Script", lines=4)
            fwd_btn = gr.Button("Convert to Telugu →", variant="primary")
            fwd_btn.click(fn=to_telugu, inputs=fwd_in, outputs=fwd_out)
            fwd_in.submit(fn=to_telugu, inputs=fwd_in, outputs=fwd_out)
            gr.Examples(examples=FWD_EXAMPLES, inputs=fwd_in, outputs=fwd_out, fn=to_telugu)
            with gr.Accordion("Tenglish typing reference", open=False):
                gr.Markdown(REFERENCE)

        with gr.Tab("Telugu → Tenglish"):
            gr.Markdown("Paste Telugu Unicode script and get Romanized Tenglish output.")
            with gr.Row():
                rev_in = gr.Textbox(
                    label="Telugu Script",
                    placeholder="నేను తెలుగు మాట్లాడతాను",
                    lines=4,
                )
                rev_out = gr.Textbox(label="Tenglish", lines=4)
            rev_btn = gr.Button("Convert to Tenglish →", variant="primary")
            rev_btn.click(fn=to_tenglish, inputs=rev_in, outputs=rev_out)
            rev_in.submit(fn=to_tenglish, inputs=rev_in, outputs=rev_out)
            gr.Examples(examples=REV_EXAMPLES, inputs=rev_in, outputs=rev_out, fn=to_tenglish)
            gr.Markdown(
                """
**Note:** Tenglish spelling is not standardized — the same Telugu word can be validly
romanized multiple ways. This model outputs one common convention learned from training data.
                """
            )

    gr.Markdown(
        """
---
Models: [telugu-transliterator](https://huggingface.co/harinpurumandla/telugu-transliterator) (1.89% CER) ·
[telugu-to-tenglish](https://huggingface.co/harinpurumandla/telugu-to-tenglish) (16.69% CER)
        """
    )

demo.launch()
