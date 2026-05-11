"""SFT for SmolVLM2-2.2B-Instruct on the MI-Gated HatefulMemes dataset.

Assumes ``preprocess_mi_gate.py`` has already populated ``data.json`` with
``split`` and ``mi_gate_text`` fields. The user-prompt for sample n is
``mi_gate_text[n]``, which equals ``original_text + " " + caption`` when n
was selected by the MI Gate, and ``original_text`` otherwise. The assistant
target is ``correct_answer``.

Run from repo root:

    python SFT/multimodal_interaction_tuning.py --config SFT/config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from SFT.caption import build_chat_messages, load_smolvlm2  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HatefulMemesSFTDataset(Dataset):
    """Lazy dataset returning {image_path, user_text, answer} per sample.

    Images are opened in the collator to keep memory bounded.
    """

    def __init__(self, samples: List[dict], image_dir: str, text_field: str = "mi_gate_text"):
        self.samples = samples
        self.image_dir = image_dir
        self.text_field = text_field

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "image_path": os.path.join(self.image_dir, f"{s['id']}.png"),
            "user_text": str(s.get(self.text_field) or s["original_text"]),
            "answer": str(s["correct_answer"]),
            "id": int(s["id"]),
        }


def load_split(data_path: str, split: str, text_field: str) -> List[dict]:
    with open(data_path) as fh:
        data = json.load(fh)
    out = []
    for s in data:
        if s.get("split") != split:
            continue
        if not s.get(text_field) and not s.get("original_text"):
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
@dataclass
class SmolVLM2Collator:
    processor: Any
    image_dir: str
    dtype: Optional[torch.dtype] = None  # cast pixel_values to model dtype if set

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]
        user_texts = [ex["user_text"] for ex in examples]
        answers = [ex["answer"] for ex in examples]

        full_messages = [build_chat_messages(u, a) for u, a in zip(user_texts, answers)]
        prompt_messages = [build_chat_messages(u) for u in user_texts]

        full_texts = [
            self.processor.apply_chat_template(m, tokenize=False) for m in full_messages
        ]
        prompt_texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in prompt_messages
        ]

        # Tokenise full + prompt-only separately so we know exactly how many
        # tokens belong to the prompt for each sample (image-token expansion
        # is deterministic given the same image, so the prefixes match).
        image_lists = [[img] for img in images]
        full_batch = self.processor(
            text=full_texts, images=image_lists,
            return_tensors="pt", padding=True,
        )
        prompt_batch = self.processor(
            text=prompt_texts, images=image_lists,
            return_tensors="pt", padding=True,
        )

        pad_id = self.processor.tokenizer.pad_token_id
        labels = full_batch["input_ids"].clone()
        prompt_attn = prompt_batch["attention_mask"]
        for i in range(labels.shape[0]):
            prompt_len = int(prompt_attn[i].sum().item())
            labels[i, :prompt_len] = -100
        if pad_id is not None:
            labels[full_batch["input_ids"] == pad_id] = -100

        full_batch["labels"] = labels
        if self.dtype is not None and "pixel_values" in full_batch:
            full_batch["pixel_values"] = full_batch["pixel_values"].to(self.dtype)
        return dict(full_batch)


# ---------------------------------------------------------------------------
# Eval (post-training accuracy)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_accuracy(
    model,
    processor,
    samples: List[dict],
    image_dir: str,
    text_field: str,
    batch_size: int = 8,
    max_new_tokens: int = 6,
) -> Dict[str, float]:
    """Greedy generation + exact-match accuracy on Yes./No."""
    model.eval()
    device = model.device

    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"

    correct = 0
    total = 0
    by_subset = {True: [0, 0], False: [0, 0]}  # in_subset → [correct, total]

    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = [Image.open(os.path.join(image_dir, f"{s['id']}.png")).convert("RGB") for s in chunk]
        user_texts = [str(s.get(text_field) or s["original_text"]) for s in chunk]
        gold = [str(s["correct_answer"]).strip() for s in chunk]

        prompt_messages = [build_chat_messages(u) for u in user_texts]
        prompt_texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                        for m in prompt_messages]
        inputs = processor(
            text=prompt_texts, images=[[img] for img in images],
            return_tensors="pt", padding=True,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

        for s, pred, g in zip(chunk, decoded, gold):
            pred_norm = pred.strip().lower().rstrip(".")
            gold_norm = g.lower().rstrip(".")
            ok = pred_norm.startswith(gold_norm)
            correct += int(ok)
            total += 1
            in_subset = bool(s.get("mi_gate_in_subset", False))
            by_subset[in_subset][0] += int(ok)
            by_subset[in_subset][1] += 1

    metrics = {"accuracy": correct / max(1, total), "n": float(total)}
    for k, (c, t) in by_subset.items():
        if t > 0:
            metrics[f"accuracy_in_subset={k}"] = c / t
            metrics[f"n_in_subset={k}"] = float(t)
    processor.tokenizer.padding_side = original_padding_side
    return metrics


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "data_path": "data/data.json",
    "image_dir": "data/images",
    "text_field": "mi_gate_text",
    "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "output_dir": "SFT/runs/mi_gate",
    "tau": 0.25,  # informational only; gate is applied in preprocess
    "epochs": 3,
    "train_batch_size": 1,
    "eval_batch_size": 4,
    "grad_accum_steps": 8,             # effective batch = 8
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "eval_steps": None,                # default: per-epoch eval (see below)
    "save_steps": None,                # default: per-epoch save
    "save_total_limit": 2,
    "bf16": True,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 4,
    "seed": 42,
    "max_train_samples": None,
    "max_eval_samples": None,
    "eval_after_training": True,
    "eval_max_new_tokens": 6,
    "report_to": "none",

    # ---- LoRA (PEFT) ----
    "use_lora": True,                  # set False to fall back to full FT
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_bias": "none",
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_task_type": "CAUSAL_LM",
}


_PATH_KEYS = ("data_path", "image_dir", "output_dir")


def _resolve_path(value: Any) -> Any:
    """Anchor relative path strings to REPO_ROOT so CWD doesn't matter."""
    if isinstance(value, str) and value and not os.path.isabs(value):
        return os.path.join(REPO_ROOT, value)
    return value


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path) as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg.update(user_cfg)
    for key in _PATH_KEYS:
        cfg[key] = _resolve_path(cfg.get(key))
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MI-Gate SFT for SmolVLM2")
    parser.add_argument("--config", default="SFT/config.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for kv in args.overrides:
        if "=" not in kv:
            raise ValueError(f"Bad override {kv!r}; expected key=value.")
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    for k in _PATH_KEYS:
        cfg[k] = _resolve_path(cfg.get(k))

    print("[config]")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # ---------------- Data ----------------
    train_samples = load_split(cfg["data_path"], "train", cfg["text_field"])
    val_samples = load_split(cfg["data_path"], "val", cfg["text_field"])
    if cfg.get("max_train_samples"):
        train_samples = train_samples[: int(cfg["max_train_samples"])]
    if cfg.get("max_eval_samples"):
        val_samples = val_samples[: int(cfg["max_eval_samples"])]
    print(f"[data] train={len(train_samples)} val={len(val_samples)}")

    train_ds = HatefulMemesSFTDataset(train_samples, cfg["image_dir"], cfg["text_field"])
    val_ds = HatefulMemesSFTDataset(val_samples, cfg["image_dir"], cfg["text_field"])

    # ---------------- Model ----------------
    model, processor = load_smolvlm2(model_id=cfg["model_id"])
    processor.tokenizer.padding_side = "right"
    model.config.use_cache = False

    if cfg.get("use_lora", True):
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            task_type=cfg["lora_task_type"],
            r=int(cfg["lora_r"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            bias=cfg["lora_bias"],
            target_modules=list(cfg["lora_target_modules"]),
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    if cfg["gradient_checkpointing"]:
        # When using LoRA the base is frozen — make sure input embeddings still
        # produce grads so they propagate through to the adapters.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # PEFT wraps the model; pull dtype from a base parameter.
    model_dtype = next(model.parameters()).dtype
    collator = SmolVLM2Collator(processor=processor, image_dir=cfg["image_dir"], dtype=model_dtype)

    # ---------------- Trainer ----------------
    from transformers import Trainer, TrainingArguments

    # Eval/save strategy: per-epoch when steps are null, per-step otherwise.
    if cfg.get("eval_steps"):
        eval_strategy = "steps"
        eval_steps_arg = int(cfg["eval_steps"])
    else:
        eval_strategy = "epoch"
        eval_steps_arg = None
    if cfg.get("save_steps"):
        save_strategy = "steps"
        save_steps_arg = int(cfg["save_steps"])
    else:
        save_strategy = "epoch"
        save_steps_arg = None
    # load_best_model_at_end requires matching strategies; we use the same.
    load_best = (eval_strategy == save_strategy) and val_samples and len(val_samples) > 0

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["train_batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        eval_strategy=eval_strategy,
        eval_steps=eval_steps_arg,
        save_strategy=save_strategy,
        save_steps=save_steps_arg,
        save_total_limit=cfg["save_total_limit"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg["gradient_checkpointing"] else None,
        dataloader_num_workers=cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        ddp_find_unused_parameters=False,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False if load_best else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(cfg["output_dir"])

    # ---------------- Post-training accuracy on val ----------------
    if cfg["eval_after_training"]:
        print("[eval] generation-based accuracy on val split")
        metrics = evaluate_accuracy(
            model=model,
            processor=processor,
            samples=val_samples,
            image_dir=cfg["image_dir"],
            text_field=cfg["text_field"],
            batch_size=cfg["eval_batch_size"],
            max_new_tokens=cfg["eval_max_new_tokens"],
        )
        print("[eval] metrics:", metrics)
        with open(os.path.join(cfg["output_dir"], "eval_metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)


if __name__ == "__main__":
    main()
