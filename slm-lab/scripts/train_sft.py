"""Phase-1: SFT + LoRA on Qwen3-0.6B for the Intent Router task.

Reads configs/sft.yaml. Uses TRL's SFTTrainer with a PEFT LoRA config. The
dataset is conversational (messages), so SFTTrainer applies the model's chat
template automatically.

Usage:
    python scripts/train_sft.py --config configs/sft.yaml
"""

import argparse

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = load_dataset(
        "json",
        data_files={
            "train": cfg["data"]["train_file"],
            "validation": cfg["data"]["valid_file"],
        },
    )

    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    t = cfg["training"]
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 100),
        warmup_ratio=t.get("warmup_ratio", 0.0),
        lr_scheduler_type=t.get("lr_scheduler_type", "linear"),
        bf16=t.get("bf16", False),
        max_length=t.get("max_length", 1024),
        packing=t.get("packing", False),
        report_to=t.get("report_to", "none"),
        load_best_model_at_end=t.get("load_best_model_at_end", False),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=t.get("greater_is_better", False),
        save_total_limit=t.get("save_total_limit", None),
    )

    callbacks = []
    patience = t.get("early_stopping_patience")
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = SFTTrainer(
        model=cfg["model_name"],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Saved LoRA adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
