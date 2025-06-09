import re
import unicodedata

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Schemes to try in order — ITRANS is closest to natural Tenglish spelling
SCHEMES = [sanscript.ITRANS, sanscript.HK, sanscript.VELTHUIS]

_MULTI_SPACE = re.compile(r"\s+")


def _clean_input(text: str) -> str:
    return _multi_space_clean(unicodedata.normalize("NFC", text).strip())


def _multi_space_clean(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text).strip()


def rule_based_transliterate(roman_text: str) -> str:
    cleaned = _clean_input(roman_text)
    # Try each scheme — use whichever produces the most Telugu characters
    best = ""
    best_te_ratio = 0.0
    for scheme in SCHEMES:
        try:
            result = transliterate(cleaned, scheme, sanscript.TELUGU)
            result = _multi_space_clean(result)
            if not result:
                continue
            te_chars = sum(1 for c in result if 0x0C00 <= ord(c) <= 0x0C7F)
            ratio = te_chars / len(result) if result else 0.0
            if ratio > best_te_ratio:
                best_te_ratio = ratio
                best = result
        except Exception:
            continue
    return best if best else roman_text


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    m, n = len(ref), len(pred)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if ref[i-1] == pred[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n] / m


def evaluate_baseline(pairs: list[dict]) -> dict:
    total_cer = 0.0
    exact = 0
    for pair in pairs:
        roman = pair.get("roman_text", "")
        ref = pair.get("telugu_text", "")
        pred = rule_based_transliterate(roman)
        total_cer += cer(pred, ref)
        if pred.strip() == ref.strip():
            exact += 1
    n = len(pairs)
    return {
        "model": "rule_based_itrans",
        "count": n,
        "cer": round(total_cer / n, 4) if n else 0.0,
        "exact_match": round(exact / n, 4) if n else 0.0,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        ("nenu", "నేను"),
        ("ela unnav", "ఎలా ఉన్నావ్"),
        ("vastanu", "వస్తాను"),
        ("dhanyavallu ra", "ధన్యవాదాలు రా"),
        ("super ga undi", "సూపర్ గా ఉంది"),
        ("nenu vastunna bro", "నేను వస్తున్న బ్రో"),
        ("cheppu da", "చెప్పు దా"),
    ]
    print(f"{'Roman':<25} {'Rule-based output':<30} {'CER':>6}")
    print("-" * 65)
    for roman, ref in tests:
        pred = rule_based_transliterate(roman)
        c = cer(pred, ref)
        print(f"{roman:<25} {pred:<30} {c:>6.2f}")
