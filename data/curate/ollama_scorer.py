import json
import logging
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:32b"
TIMEOUT = 30
PROMPT_PATH = Path("mcp/ollama_scoring_prompt.txt")

logger = logging.getLogger(__name__)


def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _parse_response(text: str) -> dict:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found")
        return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse Ollama response: %s — %s", e, text[:100])
        return {}


def score_pair(telugu_text: str, roman_text: str) -> dict:
    template = _load_prompt_template()
    prompt = template.format(telugu_text=telugu_text, roman_text=roman_text)

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        parsed = _parse_response(raw)
        if not parsed:
            return _rules_fallback(reason="empty_response")
        return {
            "ollama_score": float(parsed.get("confidence_score", 0.5)),
            "ollama_reason": parsed.get("reason_code", "unknown"),
            "ollama_flag": bool(parsed.get("flag_for_review", False)),
            "ollama_model": MODEL,
        }
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama unavailable — falling back to rules-only mode")
        return _rules_fallback(reason="ollama_unavailable")
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out")
        return _rules_fallback(reason="timeout")


def _rules_fallback(reason: str) -> dict:
    return {
        "ollama_score": None,
        "ollama_reason": reason,
        "ollama_flag": False,
        "ollama_model": None,
    }


def score_review_bucket(review_path: Path, output_path: Path, max_pairs: int = 500) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = ollama_ok = fallback = 0

    with review_path.open(encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= max_pairs:
                break
            line = line.strip()
            if not line:
                continue
            processed += 1
            pair = json.loads(line)
            result = score_pair(
                pair.get("telugu_text", ""),
                pair.get("roman_text", ""),
            )
            if result.get("ollama_score") is not None:
                ollama_ok += 1
            else:
                fallback += 1
            fout.write(json.dumps({**pair, **result}, ensure_ascii=False) + "\n")

    return {
        "processed": processed,
        "ollama_scored": ollama_ok,
        "rules_fallback": fallback,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    review_dir = Path("data/review")
    review_files = sorted(review_dir.glob("review_*.jsonl"))

    if not review_files:
        print("No review bucket files found in data/review/")
        return

    for review_file in review_files:
        out = review_dir / review_file.name.replace("review_", "ollama_scored_")
        print(f"Scoring {review_file.name} with Ollama ...")
        result = score_review_bucket(review_file, out)
        print(f"  {result['processed']} pairs: "
              f"{result['ollama_scored']} ollama, {result['rules_fallback']} fallback")


if __name__ == "__main__":
    main()
