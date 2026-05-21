"""Gradio demo for TeluguTransliterator transliteration."""

import re as _re

import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "harinpurumandla/telugu-transliterator"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()


# Matra followed by the same standalone vowel is a model artifact — strip the redundant vowel.
# e.g. రాఆ → రా, రీఈ → రీ
_MATRA_ARTIFACT = [
    ("ాఆ", "ా"), ("ిఇ", "ి"), ("ీఈ", "ీ"), ("ుఉ", "ు"), ("ూఊ", "ూ"),
    ("ెఎ", "ె"), ("ేఏ", "ే"), ("ొఒ", "ొ"), ("ోఓ", "ో"),
]

# Tenglish trailing 'y' on 'e' endings: "antey"→"ante" before inference.
_EY_SUFFIX = _re.compile(r'(?<=[a-z])ey\b', _re.IGNORECASE)


def _fix_artifacts(text: str) -> str:
    for bad, good in _MATRA_ARTIFACT:
        text = text.replace(bad, good)
    return text


def _normalize_tenglish(text: str) -> str:
    return _EY_SUFFIX.sub('e', text)


def transliterate(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = _normalize_tenglish(text)
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=1)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _fix_artifacts(result)


EXAMPLES = [
    ["nenu Telugu maatladutaanu"],
    ["ela unnav bro"],
    ["pakka cheppindi"],
    ["ikkade unna"],
    ["super ga undi"],
    ["em chestunnav"],
    ["konchem wait cheyyi"],
    ["nenu school ki vellanu"],
    ["anthey nuvvu cheppindi correct"],
    ["sare raa intiki"],
]

VOWEL_TABLE = """
### Vowels

| How to type | Telugu | Sound | Example |
|---|---|---|---|
| a | అ | short a | anu అను |
| aa | ఆ | long aa | aame ఆమె |
| i | ఇ | short i | ika ఇక |
| ee / ii | ఈ | long ee | eedhi ఈది |
| u | ఉ | short u | uku ఉకు |
| uu / oo | ఊ | long oo | uuru ఊరు |
| e | ఎ | short e | ela ఎలా |
| ee / ae | ఏ | long ae | eela ఏల |
| ai | ఐ | ai sound | aithe ఐతే |
| o | ఒ | short o | okka ఒక్క |
| oh / O | ఓ | long oh | oka ఓక |
| au / ow | ఔ | au sound | aunu ఔను |
"""

CONSONANT_TABLE = """
### Key Consonant Pairs (most common confusion)

| Type | Tenglish | Telugu | Example |
|---|---|---|---|
| Retroflex T | t | ట | antey అంటే |
| Dental T | th | త / ంత | anthey అంతే |
| Retroflex D | d | డ | adda అడ్డ |
| Dental D | dh | ద | dhaari దారి |
| Retroflex N | N / nn | ణ | paNi పణి |
| Dental N | n | న | nenu నేను |
| Retroflex L | L / ll | ళ | telLu తెళ్ళు |
| Regular L | l | ల | ledu లేదు |
| Sha | sh / S | శ | shanti శాంతి |
| Retroflex Sha | Sh | ష | pushpa పుష్ప |

### All Consonants

| Tenglish | Telugu | Tenglish | Telugu | Tenglish | Telugu |
|---|---|---|---|---|---|
| k | క | kh | ఖ | g | గ |
| gh | ఘ | ch | చ | chh | ఛ |
| j | జ | jh | ఝ | t (retroflex) | ట |
| th (dental) | త | d (retroflex) | డ | dh (dental) | ద |
| n | న | p | ప | ph / f | ఫ |
| b | బ | bh | భ | m | మ |
| y | య | r | ర | l | ల |
| v / w | వ | s | స | sh | శ |
| h | హ | L / ll | ళ | z | జ |
"""

TIPS_TABLE = """
### Common Words & Patterns

| Tenglish | Telugu | Meaning |
|---|---|---|
| antey | అంటే | meaning / if you say |
| anthey | అంతే | that's all |
| ledu | లేదు | no / not there |
| undi | ఉంది | is / there is |
| unnav | ఉన్నావ్ | are you (there) |
| raa | రా | come |
| vella | వెళ్ళ | go |
| cheyyi | చెయ్యి | do it / hand |
| ela | ఎలా | how |
| em | ఏం | what |
| pakka | పక్కా | definitely (Urdu) |
| bilkul | బిల్కుల్ | absolutely (Urdu) |
| konchem | కొంచెం | a little |
| ikkade | ఇక్కడే | right here |

### Tips
- **English words** (bro, ok, super, wait) pass through unchanged — just type them normally
- **Doubled vowels** indicate length: `aa` = ఆ, `ee` = ఈ, `oo` = ఊ
- **t vs th**: use `t` for ట (antey), `th` for త/ంత (anthey)
- **Long sentences** (>40 words) may lose accuracy — split into sentences
"""

with gr.Blocks(title="TeluguTransliterator") as demo:
    gr.Markdown(
        """
# TeluguTransliterator

Convert Romanized Telugu (Tenglish) to Telugu Unicode script.
Handles chat-style spellings, code-mix English, and colloquial Urdu loanwords.

Model: [harinpurumandla/telugu-transliterator](https://huggingface.co/harinpurumandla/telugu-transliterator)
        """
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Tenglish (Romanized Telugu)",
            placeholder="nenu Telugu maatladutaanu",
            lines=3,
        )
        out = gr.Textbox(label="Telugu Script", lines=3)

    btn = gr.Button("Transliterate", variant="primary")
    btn.click(fn=transliterate, inputs=inp, outputs=out)
    inp.submit(fn=transliterate, inputs=inp, outputs=out)

    gr.Examples(examples=EXAMPLES, inputs=inp, outputs=out, fn=transliterate)

    with gr.Accordion("How to type — Tenglish reference card", open=False):
        gr.Markdown(VOWEL_TABLE)
        gr.Markdown(CONSONANT_TABLE)
        gr.Markdown(TIPS_TABLE)

demo.launch()
