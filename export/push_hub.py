"""Push trained model and tokenizer to HuggingFace Hub."""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


REPO_ID = "harinpurumandla/telugu-transliterator"
DATASET_REPO_ID = "harinpurumandla/telugu-transliterator-dataset"


def push_model(checkpoint_dir: str, model_card: str, dry_run: bool = False) -> None:
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        print(f"Checkpoint not found: {checkpoint_dir}")
        sys.exit(1)

    print(f"Loading model from {checkpoint_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt))

    if dry_run:
        print("Dry run — skipping hub push.")
        print(f"  Would push to: {REPO_ID}")
        print(f"  Model card:    {model_card}")
        return

    print(f"Pushing model to {REPO_ID} ...")
    tokenizer.push_to_hub(REPO_ID)
    model.push_to_hub(REPO_ID)

    if Path(model_card).exists():
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"Model card uploaded from {model_card}")

    print(f"Done. Model available at https://huggingface.co/{REPO_ID}")


def main() -> None:
    global REPO_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="train/checkpoints_v3/checkpoint-80000")
    parser.add_argument("--model-card", default="MODEL_CARD.md")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    REPO_ID = args.repo_id
    push_model(args.checkpoint, args.model_card, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
