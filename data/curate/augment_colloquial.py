import json
import re
import uuid
from pathlib import Path

# ─── Tier 1 — Very high frequency (apply to ~50 % of pairs) ──────────────────
# These variants appear in essentially all Telugu chat corpora.
# Sources: Dakshina (LREC 2020), Aksharantar (EMNLP 2023), RANLP 2021 code-mix,
#          Kirov et al. CL 2024 k-best sampling study.

TIER1_RULES: list[tuple[str, list[str]]] = [
    # Long-vowel shortening — the single most pervasive informal pattern
    (r"aa", ["a"]),
    (r"ee", ["e", "i"]),
    (r"oo", ["o", "u"]),
    (r"uu", ["u"]),

    # Aspiration drop — aspirates are Sanskrit-origin; native Telugu phonology
    # is weakly aspirated, so chat writers routinely drop the h
    (r"(?<![a-z])kh", ["k"]),
    (r"(?<![a-z])gh", ["g"]),
    (r"bh", ["b"]),
    (r"ph", ["p", "f"]),
    (r"(?<=[aeiou])th", ["t"]),     # medial dental-th
    (r"(?<=[aeiou])dh", ["d"]),     # medial dh

    # Geminate consonant simplification — retroflexes and all geminates often
    # written as singles in fast chat (ACM TALIP 2023; RANLP 2021)
    (r"kk", ["k"]),
    (r"tt", ["t"]),
    (r"dd", ["d"]),
    (r"pp", ["p"]),
    (r"mm", ["m"]),
    (r"nn", ["n"]),
    (r"ll", ["l"]),
    (r"ss", ["s"]),
    (r"cc", ["c"]),
    (r"rr", ["r"]),

    # v / w word-initial interchange — dialectal (Hyderabad v, diaspora w)
    (r"\bv", ["w"]),
    (r"\bw", ["v"]),

    # Final short-u drop after consonant — very common in fast typing
    # e.g. vaadu → vaad, vellu → vell
    (r"([bcdfghjklmnpqrstvwxyz])u\b", [r"\1"]),
]

# ─── Tier 2 — Medium frequency (apply to ~20 % of pairs) ─────────────────────

TIER2_RULES: list[tuple[str, list[str]]] = [
    # Diphthong variants
    (r"\bai\b", ["ay", "aay", "ae"]),
    (r"ai\b", ["ay", "aay"]),
    (r"au\b", ["ow", "avu"]),
    (r"\bau", ["ow", "av"]),

    # Fricative merger: শ (sha) and ষ (Sha) both → sh or s in informal writing
    (r"sh", ["s"]),

    # ch / c shortening for చ
    (r"\bch", ["c"]),

    # Retroflex-dental merger for words that preserve the distinction with doubled
    # consonants — capital-letter ITRANS convention abandoned in chat
    (r"\bT([aeiou])", [r"t\1"]),
    (r"\bD([aeiou])", [r"d\1"]),

    # Dative suffix — ki / ku are both attested grammatical variants
    # -ki more Andhra, -ku more Telangana/archaic; freely interchangeable in chat
    (r"ku\b", ["ki"]),
    (r"ki\b", ["ku", "ke"]),

    # Accusative -ni / -nu variation
    (r"ni\b", ["nu"]),
    (r"nu\b", ["ni", "n"]),

    # Locative -lo / -la (Telangana dialect influence)
    (r"\blo\b", ["la"]),
    (r"\bla\b", ["lo"]),

    # Sociative / instrumental -to variants: thoo, too, tho
    (r"to\b", ["thoo", "too", "tho"]),

    # Adverbial -gaa / -laa shortening
    (r"gaa\b", ["ga"]),
    (r"laa\b", ["la"]),
    (r"raa\b", ["ra"]),

    # Continuous-tense shortening: -tunnaanu → -tunna → -tuna
    (r"tunnaanu\b", ["tunna", "tuna"]),
    (r"tunnav\b", ["tunav", "tunaav"]),
    (r"tunnaav\b", ["tunnav", "tunav"]),

    # Past-tense -aanu shortening
    (r"aanu\b", ["anu", "nu"]),

    # 3rd-person past: -aaDu / -adu, -indi / -undi
    (r"aaDu\b", ["adu", "aadu"]),
    (r"indi\b", ["undi"]),
    (r"undi\b", ["indi"]),

    # Negative -ledu variants
    (r"\bledu\b", ["le", "ledhu"]),
    (r"\bledhu\b", ["ledu", "le"]),

    # -nu suffix shortening (command / conditional)
    (r"nu\b", ["n"]),

    # ante / anthe / antey conflation — two distinct Telugu words (అంటే / అంతే)
    # written identically in informal Tenglish (LREC 2022; RANLP 2021)
    (r"\bante\b", ["anthe", "antey", "anthey"]),
    (r"\banthe\b", ["ante", "antey", "anthey"]),
    (r"\bantey\b", ["ante", "anthe", "anthey"]),
    (r"\banthey\b", ["ante", "anthe", "antey"]),

    # Nasal interchange before labials / velars
    (r"am([bmp])", [r"an\1"]),
    (r"an([bmp])", [r"am\1"]),

    # Nasal coda: word-final -m → -n or dropped (vijayam → vijayan)
    (r"am\b", ["an", "um"]),
    (r"em\b", ["im", "an"]),

    # Word-final -ra shortening
    (r"ra\b", ["r"]),
]

# ─── Tier 3 — Low frequency / regional (apply to ~5 % of pairs) ──────────────

TIER3_RULES: list[tuple[str, list[str]]] = [
    # Telangana e → i in unstressed syllables (meeru → miru, ledu → lidu)
    (r"eru\b", ["iru"]),
    (r"elu\b", ["ilu"]),
    (r"enu\b", ["inu"]),

    # o / u neutralization in unstressed syllables (Telangana vowel harmony)
    (r"([^aeiou])o([^aeiou])", [r"\1u\2"]),

    # v / w interchange word-medially for some speakers
    (r"([aeiou])v([aeiou])", [r"\1w\2"]),
    (r"([aeiou])w([aeiou])", [r"\1v\2"]),

    # j / z word-initial for Urdu loanwords (jaroor ↔ zaroor)
    (r"\bjz", ["z"]),
    (r"\bzj", ["j"]),
    (r"\bjar", ["zar"]),
    (r"\bzar", ["jar"]),

    # ksh / x / ks interchangeability (laxmi / lakshmi / laksmi)
    (r"ksh", ["x", "ks"]),
    (r"\bx\b", ["ksh", "ks"]),

    # Emphatic vowel prolongation (ACM TALIP 2023 — stress/surprise marking)
    (r"aa\b", ["aaa"]),
    (r"oo\b", ["ooo"]),
    (r"ee\b", ["eee"]),

    # English loanword terminal-u insertion / drop
    # Telugu phonotactics disallow consonant-final words, so bus → busu; but
    # chat writers also drop the epenthetic u back: classu → class
    (r"([bcdfghjklmnpqrstvwxyz])\b", [r"\1u"]),

    # Ultra-short chat abbreviations (rarely used but attested)
    (r"\bnuvvu\b", ["nv"]),
    (r"\bnenu\b", ["ne"]),
    (r"\bthanks\b", ["tq", "thanx", "thnx"]),
    (r"\bokay\b", ["ok", "okk"]),
    (r"\bok\b", ["okay", "okk"]),

    # Urdu loanword spelling variants (Hyderabad Telugu chat)
    (r"\bpakka\b", ["pukka", "paka"]),
    (r"\bpukka\b", ["pakka", "paka"]),
    (r"\bbilkul\b", ["bilkool", "bilkull"]),
    (r"\bbilkool\b", ["bilkul"]),
    (r"\bzaroor\b", ["jaroor", "zarur"]),
    (r"\bjaroor\b", ["zaroor", "zarur"]),
    (r"\bmast\b", ["maast", "masth"]),
    (r"\bthoda\b", ["toda", "thoda"]),
    (r"\bekdum\b", ["ek dum", "ekdham"]),
]

# ─── Word-level substitutions ─────────────────────────────────────────────────
# Handles specific high-frequency confusion pairs that pattern rules cannot
# cleanly target. Bidirectional where both directions are attested.

WORD_SUBS: dict[str, list[str]] = {
    # antey / anthey / anthe / ante  — అంతే vs అంటే  (most common confusion pair)
    "antey":  ["anthey", "anthe", "ante"],
    "anthey": ["antey",  "anthe", "ante"],
    "anthe":  ["antey",  "anthey", "ante"],
    "ante":   ["antey",  "anthey", "anthe"],

    # yes — avunu / avunaa / ownu / ow
    "avunu":  ["avunaa", "avuna", "ownu", "ow"],
    "avunaa": ["avunu",  "avuna", "ownu"],
    "avuna":  ["avunu",  "avunaa", "ownu"],
    "ownu":   ["avunu",  "avunaa"],
    "ow":     ["avunu",  "avunaa"],

    # here — ikkada / ikkade
    "ikkada": ["ikkade", "ikkaDa"],
    "ikkade": ["ikkada", "ikkaDa"],
    "ikkaDa": ["ikkada", "ikkade"],

    # where — ekkada / ekkade  (confusion with ikkada is common in corpus)
    "ekkada": ["ekkade", "akkada"],
    "ekkade": ["ekkada"],
    "akkada": ["akkade"],  # there
    "akkade": ["akkada"],

    # okay / alright — sare / sari / sarle
    "sare":   ["sari", "sarle", "sarlee"],
    "sari":   ["sare", "sarle"],
    "sarle":  ["sare", "sari"],
    "sarlee": ["sare", "sari", "sarle"],

    # no / not there — ledu / le / ledhu
    "ledu":   ["le", "ledhu", "leeDu"],
    "ledhu":  ["ledu", "le"],
    "le":     ["ledu", "ledhu"],

    # there is / it is — undi / undhi
    "undi":   ["undhi", "wundi"],
    "undhi":  ["undi"],
    "wundi":  ["undi"],

    # to me — naaku / naku  (reverse-model priority: naki is wrong for నాకు)
    "naaku":  ["naku"],
    "naku":   ["naaku"],

    # I — nenu / neenu
    "nenu":   ["neenu"],
    "neenu":  ["nenu"],

    # you — nuvvu / nuvu
    "nuvvu":  ["nuvu", "nuvvuu"],
    "nuvu":   ["nuvvu"],

    # is not — kaadu / kadu
    "kaadu":  ["kadu", "kaadhu"],
    "kadu":   ["kaadu"],
    "kaadhu": ["kaadu", "kadu"],

    # come (imperative) — raa / ra
    "raa":    ["ra"],
    "ra":     ["raa"],

    # a little — konchem / kunchem / koncem
    "konchem":  ["kunchem", "koncem", "kuncham"],
    "kunchem":  ["konchem", "koncem"],
    "koncem":   ["konchem", "kunchem"],

    # good / fine — baagundi / bagundi
    "baagundi":  ["bagundi", "baagumdi", "bagumdi"],
    "bagundi":   ["baagundi", "baagumdi"],
    "baagumdi":  ["baagundi", "bagundi"],

    # came — vacchaanu / vachaanu / vachanu
    "vacchaanu": ["vachaanu", "vachanu"],
    "vachaanu":  ["vacchaanu", "vachanu"],
    "vachanu":   ["vacchaanu", "vachaanu"],

    # did — chesaanu / chesanu
    "chesaanu":  ["chesanu", "chessanu"],
    "chesanu":   ["chesaanu", "chessanu"],

    # don't know — teliyadu / telidu / theliyadu
    "teliyadu":  ["telidu", "theliyadu", "thelidu"],
    "telidu":    ["teliyadu", "thelidu"],
    "theliyadu": ["teliyadu", "telidu"],

    # understood — artham / ardham
    "artham":   ["ardham"],
    "ardham":   ["artham"],

    # let's meet — kaluddam / kaludam
    "kaluddam": ["kaludam", "kaluddaam"],
    "kaludam":  ["kaluddam"],

    # tell (imperative) — cheppu / chepu
    "cheppu":  ["chepu"],
    "chepu":   ["cheppu"],

    # look / see — choodu / chudu / choosu
    "choodu":  ["chudu", "choosu"],
    "chudu":   ["choodu"],
    "choosu":  ["choodu", "chudu"],

    # talk — maatladutaanu / matladatanu
    "maatladutaanu": ["matladatanu", "maatladutanu"],
    "matladatanu":   ["maatladutaanu"],

    # going — velthunnanu / veltunna / velthunna
    "velthunnanu": ["veltunna", "velthunna", "veltuna"],
    "veltunna":    ["velthunnanu", "velthunna"],
    "velthunna":   ["velthunnanu", "veltunna"],

    # definitely / solid — pakka / pukka
    "pakka":  ["pukka", "paka", "pakaa"],
    "pukka":  ["pakka", "paka"],

    # thanks
    "thanks": ["thanx", "thnx", "tanks"],
    "thanx":  ["thanks", "tanks"],
    "tanks":  ["thanks", "thanx"],

    # okay / ok
    "okay":   ["ok", "okey", "okk"],
    "ok":     ["okay", "okey"],
    "okey":   ["ok", "okay"],
}

# Pre-compile all rule sets
_COMPILED_T1 = [(re.compile(p), reps) for p, reps in TIER1_RULES]
_COMPILED_T2 = [(re.compile(p), reps) for p, reps in TIER2_RULES]
_COMPILED_T3 = [(re.compile(p), reps) for p, reps in TIER3_RULES]


def _apply_word_subs(roman: str) -> list[str]:
    variants: list[str] = []
    words = roman.split()
    for i, word in enumerate(words):
        subs = WORD_SUBS.get(word.lower())
        if subs:
            for sub in subs[:2]:
                new_words = words[:i] + [sub] + words[i + 1:]
                variant = " ".join(new_words)
                if variant != roman:
                    variants.append(variant)
    return variants


def _apply_tier(roman: str, compiled: list, max_new: int) -> list[str]:
    found: list[str] = []
    for pattern, replacements in compiled:
        if pattern.search(roman):
            for rep in replacements[:2]:
                variant = pattern.sub(rep, roman, count=1)
                if variant != roman and variant.strip():
                    found.append(variant.strip())
        if len(found) >= max_new:
            break
    return found


def generate_variants(roman: str, max_variants: int = 5) -> list[str]:
    seen: set[str] = set()

    def _add(candidates: list[str]) -> None:
        for v in candidates:
            if v != roman and v not in seen:
                seen.add(v)

    # Tier 1 first — highest-frequency patterns
    _add(_apply_tier(roman, _COMPILED_T1, max_variants))

    # Fill remaining slots with tier 2
    if len(seen) < max_variants:
        _add(_apply_tier(roman, _COMPILED_T2, max_variants - len(seen)))

    # Word-level substitutions
    if len(seen) < max_variants:
        _add(_apply_word_subs(roman))

    # Tier 3 last — low-frequency / regional
    if len(seen) < max_variants:
        _add(_apply_tier(roman, _COMPILED_T3, max_variants - len(seen)))

    return list(seen)[:max_variants]


def augment_snapshot(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = augmented = skipped = 0

    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            pair = json.loads(line)
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")

            roman = pair.get("roman_text")
            telugu = pair.get("telugu_text")
            if not roman or not telugu:
                skipped += 1
                continue

            # Only augment approved high-confidence pairs
            if pair.get("review_status") != "approved" or pair.get("confidence", 0) < 0.6:
                continue

            variants = generate_variants(roman)
            for variant in variants:
                if variant == roman:
                    continue
                new_pair = {
                    **pair,
                    "pair_id": str(uuid.uuid4()),
                    "roman_text": variant,
                    "pair_source": "augmented",
                    "quality_score": round(pair.get("quality_score", 0.7) * 0.9, 4),
                    "confidence": round(pair.get("confidence", 0.7) * 0.9, 4),
                    "review_status": "approved",
                    "augmentation_variant": roman,
                }
                fout.write(json.dumps(new_pair, ensure_ascii=False) + "\n")
                augmented += 1

    return {
        "stage": "augment_colloquial",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "original_pairs": total,
        "augmented_pairs": augmented,
        "skipped": skipped,
    }


def main() -> None:
    interim_dir = Path("data/interim")
    inputs = sorted(interim_dir.glob("romanized_*.jsonl"))

    if not inputs:
        print("No romanized pair files found in data/interim/")
        return

    for inp in inputs:
        out = interim_dir / inp.name.replace("romanized_", "augmented_")
        print(f"Augmenting {inp.name} ...")
        result = augment_snapshot(inp, out)
        total_out = result["original_pairs"] + result["augmented_pairs"]
        print(f"  {result['original_pairs']:,} original + {result['augmented_pairs']:,} augmented = {total_out:,} total")


if __name__ == "__main__":
    main()
