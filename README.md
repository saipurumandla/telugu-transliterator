# telugu-transliterator

Data ingestion pipeline for converting Romanized Telugu (Tenglish) to Telugu script.

## Requirements

Python 3.11+
See requirements.txt

    pip install -r requirements.txt

## Usage

    python data/ingest/ingest_dakshina.py
    python data/ingest/ingest_licensed_corpus.py

Raw snapshots go to data/raw/. Pull manifests go to data/manifests/.
