import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
LABEL_QUESTIONS = {
    "World": "What is happening in world politics and international affairs?",
    "Sports": "What happened in sports events, teams, and athletes?",
    "Business": "What is happening in markets, companies, and finance?",
    "Sci/Tech": "What is happening in science and technology innovation?",
}


def make_prompt(question: str, document: str, target: str) -> str:
    return (
        "You are a document relevance judge for RAG.\n"
        "Given a user question and one news document, output only one label: "
        "relevant or not_relevant.\n\n"
        f"Question: {question}\n\n"
        f"Document: {document}\n\n"
        f"Answer: {target}"
    )


def build_relevance_pairs(num_pairs: int, seed: int) -> Dataset:
    random.seed(seed)
    ds = load_dataset("ag_news")["train"]
    df = pd.DataFrame(ds)
    df["label_name"] = df["label"].map(LABEL_MAP)

    half = num_pairs // 2
    positives: List[Dict[str, str]] = []
    negatives: List[Dict[str, str]] = []

    pos_rows = df.sample(n=half, random_state=seed).reset_index(drop=True)
    for _, row in pos_rows.iterrows():
        label = row["label_name"]
        q = LABEL_QUESTIONS[label]
        d = f"Category: {label}\nNews Article: {row['text']}"
        positives.append({"text": make_prompt(q, d, "relevant")})

    for i in range(half):
        anchor = pos_rows.iloc[i]
        true_label = anchor["label_name"]
        q = LABEL_QUESTIONS[true_label]
        wrong_pool = df[df["label_name"] != true_label]
        neg_row = wrong_pool.sample(n=1, random_state=seed + i).iloc[0]
        d = f"Category: {neg_row['label_name']}\nNews Article: {neg_row['text']}"
        negatives.append({"text": make_prompt(q, d, "not_relevant")})

    rows = positives + negatives
    random.shuffle(rows)
    return Dataset.from_list(rows)


def tokenize_dataset(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tok(batch):
        tok = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    return ds.map(_tok, batched=True, remove_columns=ds.column_names)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA relevance judge on AG News.")
    parser.add_argument(
        "--base-model-id",
        type=str,
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lora_agnews_relevance_adapter"),
    )
    parser.add_argument("--train-pairs", type=int, default=3000)
    parser.add_argument("--val-pairs", type=int, default=400)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_target_modules(model) -> List[str]:
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
    ]
    found = []
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in preferred:
            found.append(leaf)
    unique = sorted(set(found))
    if unique:
        return unique
    return ["c_attn", "c_proj"]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds = build_relevance_pairs(args.train_pairs, args.seed)
    val_ds = build_relevance_pairs(args.val_pairs, args.seed + 1)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_tok = tokenize_dataset(train_ds, tokenizer, args.max_length)
    val_tok = tokenize_dataset(val_ds, tokenizer, args.max_length)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    target_modules = infer_target_modules(model)
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_steps=40,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=default_data_collator,
    )
    trainer.train()

    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    metadata = {
        "base_model_id": args.base_model_id,
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "max_steps": args.max_steps,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
