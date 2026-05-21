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


def _eval_strategy_kwarg(train_cfg: dict) -> dict:
    import inspect
    value = train_cfg.get("eval_strategy") or train_cfg.get("evaluation_strategy", "epoch")
    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    return {key: value}


def train(config_path: str = "train/config.yaml", resume_from_checkpoint: str | None = None) -> None:
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

    offline = os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    load_kwargs = {"local_files_only": True} if offline else {}

    print(f"Loading tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], **load_kwargs)

    print("Loading datasets ...")
    train_pairs = load_jsonl(data_cfg["train_file"])
    val_pairs = load_jsonl(data_cfg["val_file"])
    print(f"  train: {len(train_pairs):,}  val: {len(val_pairs):,}")

    input_field = data_cfg.get("input_field", "roman_text")
    target_field = data_cfg.get("target_field", "telugu_text")
    print(f"Direction: {input_field} → {target_field}")

    train_dataset = TranslitDataset(
        train_pairs, tokenizer,
        model_cfg["max_input_length"],
        model_cfg["max_target_length"],
        input_field=input_field,
        target_field=target_field,
    )
    val_dataset = TranslitDataset(
        val_pairs, tokenizer,
        model_cfg["max_input_length"],
        model_cfg["max_target_length"],
        input_field=input_field,
        target_field=target_field,
    )

    print(f"Loading model: {model_cfg['name']}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg["name"], **load_kwargs)
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
        **_eval_strategy_kwarg(train_cfg),
        eval_steps=train_cfg.get("eval_steps"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_steps=train_cfg.get("save_steps"),
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        logging_steps=train_cfg["logging_steps"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        predict_with_generate=train_cfg.get("predict_with_generate", True),
        generation_max_length=train_cfg.get("generation_max_length", 128),
        # DDP: required when using gradient checkpointing with encoder-decoder models
        ddp_find_unused_parameters=False if num_gpus > 1 else None,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        total = len(predictions)
        # Subsample before decode — full val set (150k+) causes multi-hour CPU hang
        # 5k examples gives a reliable CER estimate (std error < 0.2%)
        if total > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(total, 5000, replace=False)
            predictions, labels = predictions[idx], labels[idx]
        print(f"[compute_metrics] total={total}  evaluating={len(predictions)}")
        # ByT5 can generate token IDs outside valid range — clip to actual vocab size
        # vocab_size returns 256 but len(tokenizer)=384 is the true upper bound
        predictions = np.clip(predictions, 0, len(tokenizer) - 1)
        print("[compute_metrics] decoding predictions ...")
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print("[compute_metrics] decoding labels ...")
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("[compute_metrics] computing CER ...")
        cer = compute_cer(decoded_preds, decoded_labels)
        print(f"[compute_metrics] done  CER={cer:.4f}")
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
    best_dir = Path(output_dir) / "best"
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        print(f"Training ended with: {e}")
    finally:
        print("[save] training loop exited — starting save ...")
        # Sync all DDP ranks before saving, then destroy the process group so
        # save_pretrained() runs without any distributed collective ops.
        if torch.distributed.is_initialized():
            print("[save] DDP barrier ...")
            torch.distributed.barrier()
            print("[save] destroying process group ...")
            torch.distributed.destroy_process_group()
        if trainer.is_world_process_zero():
            print(f"[save] rank 0 — saving model to {best_dir} ...")
            # Unwrap DDP wrapper to get the raw model — avoids any residual sync calls
            raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            best_dir.mkdir(parents=True, exist_ok=True)
            raw_model.save_pretrained(str(best_dir))
            print("[save] model weights written")
            tokenizer.save_pretrained(str(best_dir))
            print(f"[save] tokenizer written")
            print(f"Model saved to {best_dir}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "train/config.yaml"
    resume_checkpoint = sys.argv[2] if len(sys.argv) > 2 else None
    train(config_path, resume_from_checkpoint=resume_checkpoint)
