---
language: te
tags:
  - transliteration
  - telugu
  - tenglish
  - low-resource
license: apache-2.0
---

# TeluguTransliterator Dataset

Training dataset for converting Romanized Telugu (Tenglish) to Telugu Unicode script.
Built from publicly licensed corpora with full provenance tracking.

## Dataset Summary

| Source | Pairs | License | Type |
|--------|-------|---------|------|
| Aksharantar | 4,709,570 | CC-BY-4.0 | word-level, direct alignment |
| Samanantar | 904,342 | CC-BY-4.0 | sentence-level, parallel corpus |
| Wikipedia Telugu | 458,409 | CC-BY-SA-3.0 | sentence-level, romanized via rules |
| Dakshina | 89,134 | CC-BY-SA-4.0 | lexicon pairs |
| Synthetic (Ollama/Gemma4) | 3,421 | internal | targeted augmentation |
| **Total** | **6,164,876** | | |

## Splits

| Split | Pairs |
|-------|-------|
| train | 5,549,032 |
| val | 306,625 |
| test | 309,219 |

## Fields

Each record is a JSON line with:
```json
{
  "pair_id": "uuid",
  "roman_text": "nenu Telugu maatladutaanu",
  "telugu_text": "నేను తెలుగు మాట్లాడుతాను",
  "source_name": "aksharantar",
  "license_tag": "CC-BY-4.0",
  "pair_source": "direct",
  "quality_score": 0.9,
  "confidence": 0.9,
  "review_status": "approved",
  "created_at": "2026-05-17T...",
  "augmentation_variant": null
}
```

## Provenance

All pairs carry full provenance: source name, source document ID, license tag, pull timestamp,
and row hash of the original raw text. Immutable raw snapshots are preserved locally.

Raw data was pulled on 2026-04-05 to 2026-04-19. Pipeline stages:
normalize → filter → build_pairs → romanize → improve → augment → dedup → quality_score → split

Quality gates:
- Length ratio filter: `len(roman) / len(telugu)` must be in [0.3, 4.0]
- Telugu script sanity: ≥10% Telugu Unicode characters
- Quality score threshold: ≥0.4 for train inclusion (lower routed to review bucket)
- Near-deduplication: MinHash LSH, Jaccard threshold 0.85

## License

This dataset inherits the most restrictive license of its sources: **CC-BY-SA-4.0**.
Synthetic pairs generated via local Ollama are labeled `internal` and covered under Apache-2.0.

See `data/sources/license_matrix.yaml` for full license tier definitions.

## Citation

```
@misc{telugu-transliterator-dataset-2026,
  title={TeluguTransliterator Dataset: Curated Tenglish to Telugu Transliteration Pairs},
  author={Purumandla, Sai Harin Kumar Reddy},
  year={2026},
  url={https://huggingface.co/datasets/harinpurumandla/telugu-transliterator-dataset}
}
```
