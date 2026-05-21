"""
Synthetic Tenglish-Telugu pair generation using local Ollama.
Targets three gaps identified in Phase 3 baseline:
  1. Urdu loanwords common in Telugu chat
  2. English code-mix (bro, ok, super, sorry)
  3. Long natural Tenglish sentences

All generated records labeled with full trace metadata.
"""

import hashlib
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:31b"
PROMPT_VERSION = "v1"
SOURCE_URL = "http://localhost:11434"
LICENSE_TAG = "internal"

TELUGU_START = 0x0C00
TELUGU_END = 0x0C7F

EXAMPLES = """Examples of correct Tenglish (informal Telugu romanization):
TENGLISH: nenu chala busy ga unna da, sorry late reply ki
TELUGU: నేను చాలా బిజీగా ఉన్నా దా, సారీ లేట్ రిప్లైకి

TENGLISH: pakka ra, nenu repu vosta bro
TELUGU: పక్కా రా, నేను రేపు వస్తా బ్రో

TENGLISH: ok ok, mast ga undi, cheppu ela unnav
TELUGU: ok ok, మస్త్ గా ఉంది, చెప్పు ఎలా ఉన్నావ్

TENGLISH: nenu movie ki velladam ante, super ga undi ani cheppadu
TELUGU: నేను మూవీకి వెళ్ళడం అంటే, సూపర్ గా ఉంది అని చెప్పాడు

TENGLISH: bilkul correct cheppav bro, nenu kuda appude cheppanu
TELUGU: బిల్కుల్ కరెక్ట్ చెప్పావ్ బ్రో, నేను కూడా అప్పుడే చెప్పాను"""

PROMPTS = {
    "urdu_loanwords": {
        "count": 1000,
        "system": "You are a Telugu language expert who understands informal Telugu chat (Tenglish). Generate authentic Tenglish-Telugu pairs. Tenglish is how Telugu speakers type in Roman script on WhatsApp - informal, shortened, mixed with Urdu/English words.",
        "user": f"""{EXAMPLES}

Now generate {{n}} MORE Tenglish-Telugu pairs that include Urdu loanwords common in Telugu speech: pakka, bilkul, mast, sahi, seedha, khush, theek, accha, zabardast, tension, problem.

Rules:
- Tenglish must be informal, like real WhatsApp messages
- Mix Telugu romanized words with Urdu words naturally
- Vary length: some short (3-6 words), some longer (8-15 words)
- DO NOT write English sentences - write Telugu words in Roman script

Output ONLY the pairs, no explanation:
TENGLISH: <informal roman telugu with urdu words>
TELUGU: <proper telugu script>"""
    },
    "code_mix": {
        "count": 1000,
        "system": "You are a Telugu language expert. Generate authentic Tenglish-Telugu pairs. Tenglish mixes Roman-script Telugu with English words like ok, bro, super, nice, sorry, wait, done.",
        "user": f"""{EXAMPLES}

Generate {{n}} Tenglish-Telugu pairs where Telugu speakers naturally mix English words.

Rules:
- Keep English words (ok, bro, super, nice, sorry, thanks, wait, done, call, check) in English
- Write Telugu words in informal Roman script
- Natural WhatsApp style - short to medium length

Output ONLY:
TENGLISH: <roman telugu with english words>
TELUGU: <telugu script, english words transliterated or kept>"""
    },
    "long_sentences": {
        "count": 800,
        "system": "You are a Telugu language expert. Generate longer informal Telugu chat messages (sentences about daily life) in both Tenglish and Telugu script.",
        "user": f"""{EXAMPLES}

Generate {{n}} longer Tenglish-Telugu pairs (10-20 words) about: food, travel, family, work, movies, cricket, festivals, health.

Rules:
- Write as real WhatsApp sentences, not formal Telugu
- Drop final 'u' in casual speech (vastanu -> vastu, cheppadu -> cheppadu)
- Can include some English or Urdu words naturally
- Make them sound like real conversation

Output ONLY:
TENGLISH: <longer informal tenglish sentence>
TELUGU: <telugu script>"""
    },
}


def _row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_valid_telugu(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    te = sum(1 for c in text if TELUGU_START <= ord(c) <= TELUGU_END)
    return te / len(text) >= 0.15


def _parse_pairs(response_text: str) -> list[tuple[str, str]]:
    pairs = []
    tenglish_pat = re.compile(r'TENGLISH:\s*(.+)', re.IGNORECASE)
    telugu_pat = re.compile(r'TELUGU:\s*(.+)', re.IGNORECASE)
    lines = response_text.strip().splitlines()
    current_tenglish = None
    for line in lines:
        t_match = tenglish_pat.match(line.strip())
        te_match = telugu_pat.match(line.strip())
        if t_match:
            current_tenglish = t_match.group(1).strip()
        elif te_match and current_tenglish:
            telugu = te_match.group(1).strip()
            if current_tenglish and telugu and _is_valid_telugu(telugu):
                pairs.append((current_tenglish, telugu))
            current_tenglish = None
    return pairs


def generate_batch(system: str, user_template: str, n: int, category: str, pull_ts: str) -> list[dict]:
    user = user_template.format(n=n)
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": 0.8, "num_ctx": 4096},
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("message", {}).get("content", "")
    except Exception as e:
        print(f"  Ollama error: {e}")
        return []

    pairs = _parse_pairs(raw)
    records = []
    for roman, telugu in pairs:
        doc_id = str(uuid.uuid4())
        for text, script, paired in [(roman, "roman", telugu), (telugu, "telugu", roman)]:
            records.append({
                "source_name": "ollama_synthetic",
                "source_doc_id": f"syn_{category}_{doc_id}_{'ro' if script=='roman' else 'te'}",
                "source_url": SOURCE_URL,
                "license_tag": LICENSE_TAG,
                "pull_timestamp_utc": pull_ts,
                "text_raw": text,
                "script_hint": script,
                "lang_hint": "te",
                "row_hash": _row_hash(text),
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "category": category,
            })
    return records


def main() -> None:
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    pull_ts = datetime.now(tz=timezone.utc).isoformat()
    output_file = Path(f"data/raw/synthetic_gemma4_{today}.jsonl")
    manifest_file = Path(f"data/manifests/synthetic_gemma4_{today}_manifest.json")

    if output_file.exists():
        print(f"Snapshot {output_file} already exists — skipping.")
        sys.exit(0)

    try:
        r = requests.get("http://localhost:11434/", timeout=5)
        print(f"Ollama running, model: {MODEL}")
    except Exception:
        print("ERROR: Ollama not running.")
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    category_counts: dict[str, int] = {}
    BATCH = 15

    with output_file.open("w", encoding="utf-8") as fout:
        for category, cfg in PROMPTS.items():
            target = cfg["count"]
            written = 0
            print(f"Generating {target} pairs for '{category}' ...")
            while written < target:
                n = min(BATCH, target - written)
                records = generate_batch(cfg["system"], cfg["user"], n, category, pull_ts)
                pairs_got = len(records) // 2
                for rec in records:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += pairs_got
                total_records += len(records)
                print(f"  {written}/{target}")
            category_counts[category] = written

    manifest = {
        "run_id": str(uuid.uuid4()),
        "source_name": "ollama_synthetic",
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "pull_timestamp_utc": pull_ts,
        "record_count": total_records,
        "pair_count": total_records // 2,
        "output_file": str(output_file),
        "license_tag": LICENSE_TAG,
        "category_counts": category_counts,
    }
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Done: {total_records // 2} pairs -> {output_file}")


if __name__ == "__main__":
    main()
