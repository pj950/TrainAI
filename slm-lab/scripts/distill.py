"""Phase-3: Knowledge distillation with TRL's GKD (Generalized KD).

Unlike plain SFT on teacher text, GKD is on-policy: the student generates
outputs and the teacher supervises them at the distribution level (KL/JSD over
logits). This is the stricter distillation described in the plan.

Requires teacher and student to share the same tokenizer/vocab (both Qwen).

Usage:
    python scripts/distill.py --config configs/distill.yaml
"""

import argparse

import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from trl import GKDConfig, GKDTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/distill.yaml")
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

    teacher = AutoModelForCausalLM.from_pretrained(
        cfg["teacher_model"], torch_dtype="auto"
    )

    peft_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        target_modules=cfg["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )

    t = cfg["training"]
    training_args = GKDConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        bf16=t.get("bf16", False),
        max_length=t.get("max_length", 1024),
        lmbda=t.get("lmbda", 0.5),
        beta=t.get("beta", 0.5),
        temperature=t.get("temperature", 1.0),
        report_to=t.get("report_to", "none"),
    )

    trainer = GKDTrainer(
        model=cfg["student_model"],
        teacher_model=teacher,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Saved distilled adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
