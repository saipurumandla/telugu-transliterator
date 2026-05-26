"""
Synthetic Tenglish↔Telugu pair generation via local Ollama gemma4.

Targets quality gaps identified in v1.1 evaluation:
  - antey / anthey disambiguation (అంటే vs అంతే)
  - dative suffix ki / ku in sentence context (naaku vs naki)
  - long-vowel pairs where length is phonemically significant
  - Urdu loanwords used naturally in Hyderabad Telugu chat
  - English code-mix with Telugu morphological markers
  - Colloquial chat: greetings, questions, reactions, commands
  - Continuous-tense and negation variant forms
  - Emotional reactions and expressive phrases

Output: RawRecord JSONL to data/raw/ — same format as other ingest scripts.
Each pair writes two records: one roman, one telugu (both immutable text_raw).
Trace log written alongside for audit.

Usage:
    python -m data.ingest.ingest_synthetic_gemma4
    python -m data.ingest.ingest_synthetic_gemma4 --model gemma4:31b --target 800
    python -m data.ingest.ingest_synthetic_gemma4 --sets antey_disambiguation dative_suffix_context
"""

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─── Config ───────────────────────────────────────────────────────────────────

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL  = "http://localhost:11434/api/chat"
DEFAULT_MODEL    = "gemma4:31b"
PROMPT_VERSION   = "v2"
SOURCE_URL       = "http://localhost:11434"
LICENSE_TAG      = "internal"

DEFAULT_TARGET_PER_SET = 700
BATCH_SIZE             = 15      # pairs requested per Ollama call
MAX_RETRIES            = 3
TIMEOUT_SECS           = 180
SLEEP_BETWEEN_CALLS    = 1.2     # seconds

TELUGU_START       = 0x0C00
TELUGU_END         = 0x0C7F
MIN_TELUGU_DENSITY = 0.10
PAIR_RATIO_MIN     = 0.30
PAIR_RATIO_MAX     = 4.00
MIN_ROMAN_LEN      = 3
MAX_ROMAN_LEN      = 512

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Prompt templates ─────────────────────────────────────────────────────────
# Each entry: version (for tracking), description, system message, user template.
# {n} is replaced with the requested batch size at call time.

PROMPTS: dict[str, dict] = {

    "antey_disambiguation": {
        "version": PROMPT_VERSION,
        "description": "అంటే (means/like) vs అంతే (that's all) in context",
        "system": (
            "You are a Telugu language expert building a transliteration training dataset. "
            "Generate authentic WhatsApp-style Tenglish (informal Roman-script Telugu) paired "
            "with correct Telugu Unicode. Output only valid JSON arrays."
        ),
        "user": """Two Telugu words are commonly confused in Tenglish:
- అంటే = "means / it means / so / like that" — used for explanation or inference
- అంతే = "that's all / only that / just that / nothing more" — used for finality

Generate {n} sentence pairs. Vary between both words in context so the model sees clear examples of each.

Return ONLY a JSON array, no other text:
[{{"roman_text": "antey nuvvu raaledu anukuntunna", "telugu_text": "అంటే నువ్వు రాలేదు అనుకుంటున్న"}},
 {{"roman_text": "naaku antey chalu ee salary", "telugu_text": "నాకు అంతే చాలు ఈ శాలరీ"}},
 {{"roman_text": "anthey em chesaav ippudu", "telugu_text": "అంటే ఏం చేశావ్ ఇప్పుడు"}},
 {{"roman_text": "okasari antey ani cheppindi", "telugu_text": "ఒకసారి అంతే అని చెప్పింది"}}]

Generate {n} pairs now:""",
    },

    "dative_suffix_context": {
        "version": PROMPT_VERSION,
        "description": "Dative -ki/-ku in sentence context — naaku/naku/naaki disambiguation",
        "system": (
            "You are a Telugu language expert building a transliteration training dataset. "
            "Generate authentic WhatsApp-style Tenglish paired with correct Telugu Unicode."
        ),
        "user": """The Telugu dative case suffix appears in many forms in Tenglish:
- నాకు (naaku / naku) = to me  [NOT naki — నాకి is non-standard]
- నీకు (neeku / niku) = to you
- వాడికి (vaadiki) = to him
- అమ్మకి (ammaki) = to mother
- ఇంటికి (intiki) = towards home
- స్కూల్కి (schoolki) = to school
- ఆఫీస్కి (officeki) = to office

Generate {n} chat sentences that naturally use dative forms. Vary the subject/object.

Return ONLY a JSON array:
[{{"roman_text": "naaku aa vishayam teliyadhu", "telugu_text": "నాకు ఆ విషయం తెలియదు"}},
 {{"roman_text": "intiki ellapudu vastav", "telugu_text": "ఇంటికి ఎప్పుడు వస్తావ్"}},
 {{"roman_text": "neeku cheppali anukuntunna", "telugu_text": "నీకు చెప్పాలి అనుకుంటున్న"}},
 {{"roman_text": "ammaki phone chesaava", "telugu_text": "అమ్మకి ఫోన్ చేశావా"}}]

Generate {n} pairs now:""",
    },

    "long_vowel_disambiguation": {
        "version": PROMPT_VERSION,
        "description": "Sentences where vowel length is phonemically meaningful",
        "system": (
            "You are a Telugu language expert. Generate WhatsApp-style Tenglish paired "
            "with correct Telugu Unicode. Telugu vowel length changes word meaning."
        ),
        "user": """Telugu vowel length is meaningful — generate sentences that use these words correctly:
రాక (raaka) = coming  |  రకం (rakam) = type/kind
పాట (paaTa) = song    |  పటం (paTam) = picture/kite
నీరు (neeru) = water  |  నిర (nira) = full (archaic)
కాలు (kaalu) = leg    |  కలు (kalu) = arrack
ఊరు (ooru) = town/village  |  ఉరు (uru) = thigh
పాలు (paalu) = milk   |  పలు (palu) = several
వాడు (vaadu) = he/that guy  |  వడు (vadu) = person suffix
తీపి (teepi) = sweet  |  తిప్పు (tippu) = rotate/turn

Use these words naturally in chat sentences. Tenglish may or may not double the vowel.

Return ONLY a JSON array:
[{{"roman_text": "aa paata chala bagundi", "telugu_text": "ఆ పాట చాలా బాగుంది"}},
 {{"roman_text": "neeru teesukora", "telugu_text": "నీరు తీసుకోరా"}},
 {{"roman_text": "maa ooru super ga undi", "telugu_text": "మా ఊరు సూపర్ గా ఉంది"}},
 {{"roman_text": "vaadu ellu poyadu", "telugu_text": "వాడు ఎళ్ళిపోయాడు"}}]

Generate {n} pairs now:""",
    },

    "urdu_loanwords": {
        "version": PROMPT_VERSION,
        "description": "Hyderabad Telugu chat with Urdu loanwords naturally embedded",
        "system": (
            "You are a Telugu language expert who understands informal Hyderabad Telugu chat. "
            "Generate authentic Tenglish with Urdu loanwords paired with correct Telugu Unicode."
        ),
        "user": """Generate Hyderabad Telugu chat sentences using these Urdu-origin words naturally:
pakka = definitely/solid  |  ekdum = totally  |  bilkul = absolutely/not at all
thoda = a little          |  mast = great/awesome  |  zaroor/jaroor = certainly
seedha = direct/straight  |  zyada = too much  |  abhi = right now
tension = tension/stress  |  kaam = work  |  baad mein = later

Mix these with Telugu words naturally — as real Hyderabad speakers chat.

Return ONLY a JSON array:
[{{"roman_text": "pakka vasthanu nenu yarr", "telugu_text": "పక్కా వస్తాను నేను యార్"}},
 {{"roman_text": "thoda wait cheyyi vasthunna", "telugu_text": "తొడా వెయిట్ చెయ్యి వస్తున్న"}},
 {{"roman_text": "mast movie undi bro choodu", "telugu_text": "మస్త్ మూవీ ఉంది బ్రో చూడు"}},
 {{"roman_text": "bilkul ledu aa problem ippudu", "telugu_text": "బిల్కుల్ లేదు ఆ ప్రాబ్లెమ్ ఇప్పుడు"}},
 {{"roman_text": "ekdum correct cheppav bro", "telugu_text": "ఎక్కడం కరెక్ట్ చెప్పావ్ బ్రో"}}]

Generate {n} pairs now:""",
    },

    "code_mix_grammar": {
        "version": PROMPT_VERSION,
        "description": "English words with Telugu morphological case markers",
        "system": (
            "You are a Telugu language expert. Generate authentic code-mixed Tenglish "
            "where English nouns take Telugu grammatical markers, paired with correct Telugu Unicode."
        ),
        "user": """Telugu speakers attach Telugu case markers to English words in chat:
-ki (dative): office ki, class ki, movie ki, party ki
-lo (locative): meeting lo, bus lo, car lo, room lo
-tho (sociative): friend tho, team tho, family tho
-ni (accusative): project ni, phone ni, food ni
-tho: coffee tho, chai tho

Also Telugu verbs on English nouns: call cheyyi, text pettu, check cheyyi, update cheyyi

Generate natural-sounding sentences. Vary topics: work, college, travel, food, social plans.

Return ONLY a JSON array:
[{{"roman_text": "office ki eppudu vasthav", "telugu_text": "ఆఫీస్ కి ఎప్పుడు వస్తావ్"}},
 {{"roman_text": "meeting lo busy ga unna", "telugu_text": "మీటింగ్ లో బిజీ గా ఉన్న"}},
 {{"roman_text": "friend tho movie ki vellamu", "telugu_text": "ఫ్రెండ్ తో మూవీ కి వెళ్ళాము"}},
 {{"roman_text": "phone ni charge lo pettu", "telugu_text": "ఫోన్ ని ఛార్జ్ లో పెట్టు"}}]

Generate {n} pairs now:""",
    },

    "colloquial_chat": {
        "version": PROMPT_VERSION,
        "description": "General WhatsApp-style short conversational Telugu",
        "system": (
            "You are a Telugu language expert. Generate short, authentic WhatsApp-style "
            "Telugu conversation in both Tenglish and Telugu Unicode."
        ),
        "user": """Generate {n} authentic WhatsApp message pairs covering:
- Greetings and check-ins (ela unnav, em chestunnav)
- Questions about plans and whereabouts
- Reactions: super, darunam, bhayankaram, chala bagundi, kastam
- Requests and commands (raa, vellu, cheppu, teesuko)
- Address terms: bro, yaar, ra, maamu, anni, akka, anna
- Daily topics: food, travel, weather, family, college, cricket

Use informal Tenglish spelling — abbreviations and shortenings are fine.

Return ONLY a JSON array:
[{{"roman_text": "em tintunnav ippudu bro", "telugu_text": "ఏం తింటున్నావ్ ఇప్పుడు బ్రో"}},
 {{"roman_text": "super ga undi aa place", "telugu_text": "సూపర్ గా ఉంది ఆ ప్లేస్"}},
 {{"roman_text": "ela unnav cheppu maamu", "telugu_text": "ఎలా ఉన్నావ్ చెప్పు మామూ"}},
 {{"roman_text": "raa konchem maatladali", "telugu_text": "రా కొంచెం మాట్లాడాలి"}}]

Generate {n} pairs now:""",
    },

    "continuous_negation": {
        "version": PROMPT_VERSION,
        "description": "Continuous tense variants and negation forms",
        "system": (
            "You are a Telugu language expert. Generate WhatsApp-style Tenglish "
            "focusing on continuous tense and negation, paired with correct Telugu Unicode."
        ),
        "user": """Generate sentences using Telugu continuous tense and negation forms.

Continuous tense (all valid Tenglish spellings):
-tunnaanu / -tunna / -tuna (1st person: I am doing)
-tunnaav / -tunnav / -tunav (2nd person: you are doing)
-tunnadu / -tunadu (3rd masc: he is doing)
-tunnadi / -tunnundi (3rd fem/neuter: she/it is doing)

Negation forms:
ledu / le / ledhu = is not / not there
raadu / radu = won't / shouldn't
kaadu / kadu = is not / not that
cheyyanu / cheyyalenu = won't do
teliyadhu / teliyadu = don't know
artham kaadu / ardam kadu = don't understand

Return ONLY a JSON array:
[{{"roman_text": "nenu chestunna ee pani", "telugu_text": "నేను చేస్తున్న ఈ పని"}},
 {{"roman_text": "naaku artham kaadu em cheppav", "telugu_text": "నాకు అర్థం కాదు ఏం చెప్పావ్"}},
 {{"roman_text": "vaadu emi chestunnadu teliyadhu", "telugu_text": "వాడు ఏమి చేస్తున్నాడు తెలియదు"}},
 {{"roman_text": "adi correct kaadu bro", "telugu_text": "అది కరెక్ట్ కాదు బ్రో"}}]

Generate {n} pairs now:""",
    },

    "reactions_expressions": {
        "version": PROMPT_VERSION,
        "description": "Emotional reactions, exclamations, expressive Telugu phrases",
        "system": (
            "You are a Telugu language expert. Generate expressive WhatsApp reaction "
            "messages in Tenglish paired with correct Telugu Unicode."
        ),
        "user": """Generate expressive Telugu chat reactions and emotional phrases.

Include: ayyo, arey, arre, orey, adey, haaw — as interjections
Intensifiers: chala/chaalaa, bhayankaram, darunam, super, mast, zabardast
Emotions: surprise, excitement, frustration, humor, affection, sarcasm
Emphatic spelling is ok: chaalaaaa, superrr, nooo (appears in training data)

Return ONLY a JSON array:
[{{"roman_text": "ayyo chaalaa kastam ga undi", "telugu_text": "అయ్యో చాలా కష్టం గా ఉంది"}},
 {{"roman_text": "arey super ga chesav bro", "telugu_text": "అరే సూపర్ గా చేశావ్ బ్రో"}},
 {{"roman_text": "orey chala funny ga undi ra", "telugu_text": "ఒరే చాలా ఫన్నీ గా ఉంది రా"}},
 {{"roman_text": "bhayankaram ga undi aa scene", "telugu_text": "భయంకరం గా ఉంది ఆ సీన్"}}]

Generate {n} pairs now:""",
    },

    "long_sentences": {
        "version": PROMPT_VERSION,
        "description": "Longer colloquial sentences (10-20 words) about daily life",
        "system": (
            "You are a Telugu language expert. Generate longer informal Telugu chat sentences "
            "about daily life in both Tenglish and Telugu Unicode."
        ),
        "user": """Generate {n} longer Tenglish-Telugu pairs (10-20 words) about:
food, travel, family, work, movies, cricket, college, health, festivals, relationships.

Write as real WhatsApp messages — natural, informal, can include English or Urdu words.
Vary tenses: past, present, future, continuous.

Return ONLY a JSON array:
[{{"roman_text": "nenu chala busy ga unna bro sorry late reply ki", "telugu_text": "నేను చాలా బిజీగా ఉన్నా బ్రో సారీ లేట్ రిప్లైకి"}},
 {{"roman_text": "repu maa family tho movie ki veltunnam konchem late avutam", "telugu_text": "రేపు మా ఫ్యామిలీ తో మూవీ కి వెళ్తున్నాం కొంచెం లేట్ అవుతాం"}},
 {{"roman_text": "aa restaurant lo food chala bagundi pakka oka sari try cheyyi", "telugu_text": "ఆ రెస్టారెంట్ లో ఫుడ్ చాలా బాగుంది పక్కా ఒక సారి ట్రై చెయ్యి"}}]

Generate {n} pairs now:""",
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _telugu_density(text: str) -> float:
    if not text:
        return 0.0
    te = sum(1 for c in text if TELUGU_START <= ord(c) <= TELUGU_END)
    return te / len(text)


def _has_ascii_letter(text: str) -> bool:
    return any(c.isascii() and c.isalpha() for c in text)


def _validate(roman: str, telugu: str) -> tuple[bool, str]:
    roman  = roman.strip()
    telugu = unicodedata.normalize("NFC", telugu.strip())

    if len(roman) < MIN_ROMAN_LEN:
        return False, "roman too short"
    if len(roman) > MAX_ROMAN_LEN:
        return False, "roman too long"
    if not _has_ascii_letter(roman):
        return False, "roman has no ASCII letters"
    if sum(c.isdigit() for c in roman) / len(roman) > 0.5:
        return False, "roman >50% digits"

    density = _telugu_density(telugu)
    if density < MIN_TELUGU_DENSITY:
        return False, f"telugu density {density:.2f}"

    ratio = len(roman) / max(len(telugu), 1)
    if ratio < PAIR_RATIO_MIN or ratio > PAIR_RATIO_MAX:
        return False, f"length ratio {ratio:.2f}"

    return True, "ok"


def _parse_json_pairs(raw: str) -> list[dict]:
    """Extract JSON array from Ollama response — handles markdown fences and stray text."""
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)

    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return []

    try:
        data = json.loads(raw[start:end + 1])
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback: extract individual {...} objects
    pairs = []
    for match in re.finditer(r'\{[^{}]+\}', raw[start:end + 1]):
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                pairs.append(obj)
        except json.JSONDecodeError:
            continue
    return pairs


def _make_records(roman: str, telugu: str, category: str,
                  model: str, prompt_version: str, pull_ts: str) -> list[dict]:
    doc_id = str(uuid.uuid4())
    base = {
        "source_name": "ollama_synthetic",
        "source_url": SOURCE_URL,
        "license_tag": LICENSE_TAG,
        "pull_timestamp_utc": pull_ts,
        "lang_hint": "te",
        "model": model,
        "prompt_version": prompt_version,
        "category": category,
    }
    return [
        {**base,
         "source_doc_id": f"syn_{category}_{doc_id}_ro",
         "text_raw": roman,
         "script_hint": "roman",
         "row_hash": _row_hash(roman)},
        {**base,
         "source_doc_id": f"syn_{category}_{doc_id}_te",
         "text_raw": telugu,
         "script_hint": "telugu",
         "row_hash": _row_hash(telugu)},
    ]


# ─── Ollama call ──────────────────────────────────────────────────────────────

def _call_ollama(system: str, user: str, model: str) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.8, "top_p": 0.9, "num_ctx": 4096},
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=TIMEOUT_SECS)
            if resp.status_code == 200:
                return resp.json().get("message", {}).get("content", "")
            log.warning("Ollama HTTP %d (attempt %d)", resp.status_code, attempt)
        except requests.RequestException as exc:
            log.warning("Ollama request failed (attempt %d): %s", attempt, exc)
        if attempt < MAX_RETRIES:
            time.sleep(2 ** attempt)
    return None


# ─── Generation loop ──────────────────────────────────────────────────────────

def _load_seen_hashes(out_file: Path) -> set[str]:
    seen: set[str] = set()
    if not out_file.exists():
        return seen
    with out_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                h = d.get("row_hash")
                if h:
                    seen.add(h)
            except json.JSONDecodeError:
                pass
    return seen


def _count_pairs_per_category(out_file: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not out_file.exists():
        return counts
    with out_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                # Two records per pair — count only roman records
                if d.get("script_hint") == "roman":
                    cat = d.get("category", "unknown")
                    counts[cat] = counts.get(cat, 0) + 1
            except json.JSONDecodeError:
                pass
    return counts


def generate_for_category(
    category: str,
    cfg: dict,
    model: str,
    target: int,
    out_file: Path,
    trace_file: Path,
    seen_hashes: set[str],
    pull_ts: str,
) -> int:
    prompt_version = cfg["version"]
    generated = 0
    consecutive_empty = 0

    log.info("  [%s] target=%d", category, target)

    while generated < target:
        batch_n = min(BATCH_SIZE, target - generated + BATCH_SIZE // 2)
        user = cfg["user"].format(n=batch_n)

        raw_response = _call_ollama(cfg["system"], user, model)
        if raw_response is None:
            log.error("  [%s] Ollama unavailable — reason: ollama_timeout", category)
            _write_trace(trace_file, category, model, prompt_version, None, 0, 0,
                         ["ollama_unavailable"])
            break

        parsed = _parse_json_pairs(raw_response)
        accepted = rejected_reasons = 0
        written_records: list[dict] = []

        for raw in parsed:
            roman  = str(raw.get("roman_text",  raw.get("roman",  ""))).strip()
            telugu = str(raw.get("telugu_text", raw.get("telugu", ""))).strip()

            ok, reason = _validate(roman, telugu)
            if not ok:
                rejected_reasons += 1
                continue

            telugu = unicodedata.normalize("NFC", telugu)
            rh = _row_hash(roman)
            if rh in seen_hashes:
                rejected_reasons += 1
                continue

            records = _make_records(roman, telugu, category, model, prompt_version, pull_ts)
            written_records.extend(records)
            seen_hashes.add(rh)
            accepted += 1

        with out_file.open("a", encoding="utf-8") as f:
            for rec in written_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        _write_trace(trace_file, category, model, prompt_version,
                     raw_response, len(parsed), accepted, [f"rejected={rejected_reasons}"])

        generated += accepted
        log.info(
            "  [%s] batch: parsed=%d  accepted=%d  rejected=%d  total=%d/%d",
            category, len(parsed), accepted, rejected_reasons, generated, target,
        )

        if accepted == 0:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                log.warning("  [%s] 3 empty batches — skipping", category)
                break
        else:
            consecutive_empty = 0

        time.sleep(SLEEP_BETWEEN_CALLS)

    return generated


def _write_trace(trace_file: Path, category: str, model: str, prompt_version: str,
                 raw: str | None, parsed: int, accepted: int, notes: list[str]) -> None:
    entry = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "category": category,
        "model": model,
        "prompt_version": prompt_version,
        "raw_chars": len(raw) if raw else 0,
        "parsed": parsed,
        "accepted": accepted,
        "notes": notes,
    }
    with trace_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(model: str = DEFAULT_MODEL, target_per_set: int = DEFAULT_TARGET_PER_SET,
         sets: list[str] | None = None) -> None:
    today      = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    pull_ts    = datetime.now(tz=timezone.utc).isoformat()
    out_file   = Path(f"data/raw/synthetic_gemma4_{today}.jsonl")
    trace_file = Path(f"data/raw/synthetic_gemma4_{today}_trace.jsonl")
    manifest_file = Path(f"data/manifests/synthetic_gemma4_{today}_manifest.json")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    # Check Ollama
    try:
        ping = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in ping.json().get("models", [])]
        if model not in available:
            log.error("Model %s not in Ollama. Available: %s", model, available)
            sys.exit(1)
        log.info("Ollama ready — model: %s", model)
    except Exception as exc:
        log.error("Cannot reach Ollama at localhost:11434: %s", exc)
        sys.exit(1)

    # Resume support — load already-written hashes and per-category counts
    seen_hashes    = _load_seen_hashes(out_file)
    existing_counts = _count_pairs_per_category(out_file)
    if seen_hashes:
        log.info("Resuming — %d existing pairs loaded from %s", len(seen_hashes), out_file)

    active_sets = sets if sets else list(PROMPTS.keys())
    total_new = 0

    for category in active_sets:
        if category not in PROMPTS:
            log.warning("Unknown set: %s — skipping", category)
            continue

        already    = existing_counts.get(category, 0)
        remaining  = target_per_set - already
        if remaining <= 0:
            log.info("  [%s] complete (%d pairs) — skipping", category, already)
            continue

        n = generate_for_category(
            category=category,
            cfg=PROMPTS[category],
            model=model,
            target=remaining,
            out_file=out_file,
            trace_file=trace_file,
            seen_hashes=seen_hashes,
            pull_ts=pull_ts,
        )
        total_new += n

    final_counts = _count_pairs_per_category(out_file)
    total = sum(final_counts.values())

    manifest = {
        "run_id": str(uuid.uuid4()),
        "source_name": "ollama_synthetic",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "pull_timestamp_utc": pull_ts,
        "pair_count": total,
        "new_pairs_this_run": total_new,
        "output_file": str(out_file),
        "trace_file": str(trace_file),
        "license_tag": LICENSE_TAG,
        "category_counts": final_counts,
    }
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    log.info("─" * 60)
    log.info("Done — %d new pairs  |  %d total  |  %s", total_new, total, out_file)
    for cat, cnt in sorted(final_counts.items()):
        log.info("  %-35s  %d", cat, cnt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic Telugu pairs via local Ollama gemma4"
    )
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_PER_SET,
                        help=f"Pairs to generate per confusion set (default: {DEFAULT_TARGET_PER_SET})")
    parser.add_argument("--sets",   nargs="+", default=None,
                        choices=list(PROMPTS.keys()),
                        help="Run specific sets only (default: all)")
    args = parser.parse_args()
    main(model=args.model, target_per_set=args.target, sets=args.sets)
