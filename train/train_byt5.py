import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from train.tokenizer_utils import TranslitDataset, load_jsonl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_cer(pred_texts: list[str], label_texts: list[str]) -> float:
    total_chars = total_errors = 0
    for pred, ref in zip(pred_texts, label_texts):
        ref_chars = list(ref)
        pred_chars = list(pred)
        total_chars += len(ref_chars)
        m, n = len(ref_chars), len(pred_chars)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                if ref_chars[i - 1] == pred_chars[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        total_errors += dp[n]
    return total_errors / max(total_chars, 1)


def train(config_path: str = "train/config.yaml") -> None:
    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  {props.total_memory // 1024**3}GB")

    print(f"Loading tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])

    print("Loading datasets ...")
    train_pairs = load_jsonl(data_cfg["train_file"])
    val_pairs = load_jsonl(data_cfg["val_file"])
    print(f"  train: {len(train_pairs):,}  val: {len(val_pairs):,}")

    train_dataset = TranslitDataset(
        train_pairs, tokenizer,
        model_cfg["max_input_length"],
        model_cfg["max_target_length"],
    )
    val_dataset = TranslitDataset(
        val_pairs, tokenizer,
        model_cfg["max_input_length"],
        model_cfg["max_target_length"],
    )

    print(f"Loading model: {model_cfg['name']}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["name"])
    if train_cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    output_dir = train_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    use_bf16 = train_cfg.get("bf16", False) and torch.cuda.is_available()
    use_fp16 = train_cfg.get("fp16", False) and torch.cuda.is_available() and not use_bf16

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        seed=seed,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
        evaluation_strategy=train_cfg["evaluation_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        logging_steps=train_cfg["logging_steps"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        predict_with_generate=train_cfg.get("predict_with_generate", True),
        # DDP: required when using gradient checkpointing with encoder-decoder models
        ddp_find_unused_parameters=False if num_gpus > 1 else None,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        cer = compute_cer(decoded_preds, decoded_labels)
        return {"cer": round(cer, 4)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(f"Starting training — bf16={use_bf16}  effective_batch={train_cfg['per_device_train_batch_size'] * max(num_gpus,1)}")
    trainer.train()

    best_dir = Path(output_dir) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    print(f"Best model saved to {best_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "train/config.yaml"
    train(config_path)
