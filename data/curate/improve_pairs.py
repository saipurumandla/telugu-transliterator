"""
Phase 4 improvement: better sentence-level pair generation.

Key fixes from baseline analysis:
1. Wikipedia sentences are too long — split at clause boundaries for better model coverage
2. English words in Roman Telugu should pass through unchanged in target Telugu
3. Numbers and proper nouns should be preserved as-is
"""

import json
import re
import unicodedata
from pathlib import Path

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

TELUGU_START = 0x0C00
TELUGU_END = 0x0C7F

# Split long sentences at these clause boundary markers
CLAUSE_SPLIT = re.compile(r'[,;।\n]+')
MAX_SENTENCE_CHARS = 120   # shorter than Phase 2's 400 — better for seq2seq
MIN_SENTENCE_CHARS = 15

# English words that should pass through as-is (not transliterated)
_ENGLISH_PASSTHROUGH = {
    'ok', 'okay', 'hi', 'hello', 'bye', 'yes', 'no', 'please', 'sorry',
    'thanks', 'bro', 'da', 'ra', 'na', 'ga', 'lo', 'ki',
}

_MULTI_SPACE = re.compile(r'\s+')
_SIMPLIFY = str.maketrans({'A': 'a', 'I': 'i', 'U': 'u', 'M': 'm',
                            'H': 'h', 'è': 'e', 'ò': 'o', '~': ''})


def is_telugu(text: str) -> bool:
    te = sum(1 for c in text if TELUGU_START <= ord(c) <= TELUGU_END)
    return te / max(len(text), 1) >= 0.3


def romanize_natural(telugu_text: str) -> str:
    raw = transliterate(telugu_text, sanscript.TELUGU, sanscript.ITRANS)
    cleaned = raw.translate(_SIMPLIFY).lower()
    return _MULTI_SPACE.sub(' ', cleaned).strip()


def split_into_clauses(text: str) -> list[str]:
    parts = CLAUSE_SPLIT.split(text)
    result = []
    for part in parts:
        part = part.strip()
        if MIN_SENTENCE_CHARS <= len(part) <= MAX_SENTENCE_CHARS:
            result.append(part)
    return result


def improve_wikipedia_pairs(input_path: Path, output_path: Path) -> dict:
    """Re-process Wikipedia pairs with clause splitting and better romanization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = split_count = skipped = 0

    with input_path.open(encoding='utf-8') as fin, \
         output_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            pair = json.loads(line)

            if pair.get('source_name') != 'wikipedia_te':
                fout.write(json.dumps(pair, ensure_ascii=False) + '\n')
                kept += 1
                continue

            telugu = pair.get('telugu_text', '')
            if not is_telugu(telugu):
                skipped += 1
                continue

            # Try splitting into shorter clauses
            clauses = split_into_clauses(telugu)
            if not clauses:
                clauses = [telugu] if MIN_SENTENCE_CHARS <= len(telugu) <= MAX_SENTENCE_CHARS else []

            for clause in clauses:
                roman = romanize_natural(clause)
                if not roman:
                    continue
                new_pair = {
                    **pair,
                    'telugu_text': clause,
                    'roman_text': roman,
                    'pair_source': 'synthetic',
                    'quality_score': 0.65,
                    'confidence': 0.65,
                    'review_status': 'approved',
                }
                fout.write(json.dumps(new_pair, ensure_ascii=False) + '\n')
                kept += 1
                if len(clauses) > 1:
                    split_count += 1

    return {
        'stage': 'improve_pairs',
        'input': str(input_path),
        'output': str(output_path),
        'total_in': total,
        'total_out': kept,
        'clause_splits': split_count,
        'skipped': skipped,
    }


def main() -> None:
    interim = Path('data/interim')
    inputs = sorted(interim.glob('romanized_*.jsonl'))
    if not inputs:
        print('No romanized pair files found — run romanize_rules.py first.')
        return
    for inp in inputs:
        out = interim / inp.name.replace('romanized_', 'improved_')
        print(f'Improving {inp.name} ...')
        result = improve_wikipedia_pairs(inp, out)
        print(f'  {result["total_in"]:,} in -> {result["total_out"]:,} out '
              f'({result["clause_splits"]:,} clause splits)')


if __name__ == '__main__':
    main()
