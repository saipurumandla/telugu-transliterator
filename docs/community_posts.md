# Community Posts — TeluguTransliterator

Draft posts for sharing the model. Adapt tone per platform.

---

## Reddit — r/LanguageTechnology

**Title:** I built a Tenglish → Telugu transliteration model trained on 6M pairs — handles chat-style spelling and code-mix

**Body:**

Hey r/LanguageTechnology,

I've been building a transliteration model for Romanized Telugu (what we call "Tenglish") — the way most Telugu speakers actually type in WhatsApp, Twitter, YouTube comments, etc.

Standard transliteration tools are built for clean, formal input. They fall apart on "ela unnav bro" or "pakka cheppindi" — the kind of text that's actually out there.

**What I built:**
- Fine-tuned ByT5-small on 6.1M pairs (Aksharantar, Samanantar, Wikipedia Telugu, Dakshina + synthetic)
- Handles colloquial spellings, Urdu loanwords (pakka, bilkul, mast), English code-mix passthrough
- CER: 3.09% on colloquial slice, 7.08% overall

**Why ByT5?** Telugu Unicode fragments badly with character-level or subword tokenizers. ByT5 works at the byte level — no unknown tokens for any Telugu character, no vocabulary design decisions.

**Known limitation:** Long sentences (>40 words) hit ByT5's byte window limit — CER jumps to ~16% there.

Model: https://huggingface.co/harinpurumandla/telugu-transliterator
Dataset: https://huggingface.co/datasets/harinpurumandla/telugu-transliterator-dataset
Demo: https://huggingface.co/spaces/harinpurumandla/telugu-transliterator

Happy to discuss the data pipeline, training setup, or evaluation methodology. CER/WER numbers, confusion analysis, and the full benchmark report are in the model card.

---

## Reddit — r/telugu

**Title:** Built an AI that converts Tenglish (typing Telugu in English letters) to actual Telugu script

**Body:**

Namasthe everyone!

So you know how most of us type Telugu in WhatsApp using English letters? Like "nenu veltunna" or "ela unnav bro" — that's Tenglish.

I spent about a year building a model that converts this to actual Telugu script: "నేను వెళ్తున్నా" / "ఎలా ఉన్నావ్ bro"

It handles the usual casual typing including Urdu words we use (pakka, bilkul, super) and English words that we just leave in English.

Try it here: https://huggingface.co/spaces/harinpurumandla/telugu-transliterator

It's not perfect — long sentences can trip it up — but it works well for the kind of texts we actually send each other.

Would love to hear feedback from native speakers, especially if you find words it gets wrong!

---

## Twitter / X

**Tweet 1 (announcement):**

Built a Tenglish → Telugu transliteration model trained on 6.1M pairs 🇮🇳

Handles chat-style spelling, code-mix English, Urdu loanwords — the way Telugu speakers actually type

"ela unnav bro" → "ఎలా ఉన్నావ్ bro"
"pakka cheppindi" → "పక్కా చెప్పింది"

CER: 3.09% on colloquial text

Model + demo 👇
https://huggingface.co/harinpurumandla/telugu-transliterator

#TeluguNLP #indicnlp #NLP #transliteration #Telugu

---

**Tweet 2 (technical):**

Why ByT5 for Telugu transliteration?

Telugu Unicode has 128+ characters. Subword tokenizers split grapheme clusters in weird ways. Character-level tokenizers need big vocabs.

ByT5 works at the byte level — no vocabulary decisions, no unknown tokens for any Telugu character.

Trained on 6.1M pairs: Aksharantar + Samanantar + Wikipedia + Dakshina

#indicnlp #NLP #Telugu #ByT5

---

**Tweet 3 (the hard part):**

The hard part wasn't the model.

It was the data. Telugu text on the internet is:
- Formal Wikipedia sentences
- Chat-style abbreviations
- Mixed script (Telugu + English + Urdu loanwords)

Spent months on the curation pipeline before touching transformers.

Full dataset: https://huggingface.co/datasets/harinpurumandla/telugu-transliterator-dataset

#TeluguNLP #indicnlp #lowresource

---

## HuggingFace Community Post (Discussions tab)

**Title:** TeluguTransliterator — 6.1M pair transliteration model for colloquial Romanized Telugu

Built and releasing TeluguTransliterator — a ByT5-small model fine-tuned specifically for the way Telugu speakers actually type online.

Most transliteration models are trained on clean, formal text. Tenglish is messier: chat shortenings, code-mix English, Urdu loanwords, no consistent spelling. This model handles that.

**Highlights:**
- 6,164,876 training pairs across 5 sources (Aksharantar, Samanantar, Wikipedia, Dakshina, synthetic)
- CER 3.09% on colloquial test slice
- English code-mix passthrough preserved
- Apache-2.0 license

**Dataset:** harinpurumandla/telugu-transliterator-dataset (full provenance, license tracking per pair)
**Demo Space:** harinpurumandla/telugu-transliterator

Feedback very welcome — especially from native Telugu speakers who can judge output quality on regional or dialectal variation.

---

## IndicNLP / Mastodon / LinkedIn note

Just released TeluguTransliterator — a transliteration model for Romanized Telugu (Tenglish) trained on 6.1M pairs.

Tenglish is how most Telugu speakers type on social media: mixing spellings, English words, Urdu loanwords. Standard tools built on clean text break on this. This model is built specifically for it.

Architecture: ByT5-small (byte-level, no tokenizer vocabulary problems for Telugu script)
CER: 3.09% colloquial, 7.08% overall

Model: https://huggingface.co/harinpurumandla/telugu-transliterator
Dataset: https://huggingface.co/datasets/harinpurumandla/telugu-transliterator-dataset

#indicnlp #Telugu #NLP #transliteration #lowresource
