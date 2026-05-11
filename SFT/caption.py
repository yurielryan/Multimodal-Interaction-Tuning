"""SmolVLM2-2.2B-Instruct captioning utilities and CLI.

Reused by the SFT training script for shared model/processor loading and
chat-template construction.

CLI usage:
    python SFT/caption.py                # defaults: data/data.json + data/images
    # or with custom paths:
    python SFT/caption.py \
        --data_path /scratch/HatefulMemes/data.json \
        --image_dir /scratch/HatefulMemes/images \
        --key generated_caption_smolvlm

Library usage:
    from SFT.caption import load_smolvlm2, build_chat_messages, caption_images_batched
    model, processor = load_smolvlm2()
    captions = caption_images_batched([img1, img2], model, processor)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Optional, Sequence

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(REPO_ROOT, "data", "data.json")
DEFAULT_IMAGE_DIR = os.path.join(REPO_ROOT, "data", "images")

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
DEFAULT_CAPTION_PROMPT = (
    "Describe this meme in one or two sentences. "
    "Mention what the image depicts and quote any text shown."
)


def load_smolvlm2(
    model_id: str = DEFAULT_MODEL_ID,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    attn_implementation: Optional[str] = None,
):
    """Load SmolVLM2 model + processor on the requested device."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    kwargs = {"torch_dtype": dtype}
    if attn_implementation is not None:
        kwargs["_attn_implementation"] = attn_implementation

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model, processor


def build_chat_messages(
    user_text: str,
    assistant_text: Optional[str] = None,
) -> List[dict]:
    """Build a chat-template message list with one image and a user prompt.

    If ``assistant_text`` is provided, the assistant turn is appended (used
    during SFT). When omitted the messages are suitable for generation.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    if assistant_text is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        )
    return messages


def _apply_template(processor, messages_list: Sequence[Sequence[dict]], add_generation_prompt: bool) -> List[str]:
    return [
        processor.apply_chat_template(m, add_generation_prompt=add_generation_prompt, tokenize=False)
        for m in messages_list
    ]


@torch.inference_mode()
def caption_images_batched(
    images: Sequence[Image.Image],
    model,
    processor,
    prompt: str = DEFAULT_CAPTION_PROMPT,
    batch_size: int = 8,
    max_new_tokens: int = 96,
) -> List[str]:
    """Caption a list of PIL images using SmolVLM2.

    Forces ``padding_side='left'`` for the duration of the call (decoder-only
    generation requires it; otherwise padded sequences emit empty output) and
    restores the original setting on exit.
    """
    if not images:
        return []

    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        device = model.device
        captions: List[str] = []
        for start in range(0, len(images), batch_size):
            chunk = [img.convert("RGB") for img in images[start : start + batch_size]]
            messages_list = [build_chat_messages(prompt) for _ in chunk]
            prompts = _apply_template(processor, messages_list, add_generation_prompt=True)

            inputs = processor(
                text=prompts,
                images=[[img] for img in chunk],
                return_tensors="pt",
                padding=True,
            ).to(device)

            prompt_len = inputs["input_ids"].shape[1]
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            new_tokens = generated[:, prompt_len:]
            decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)
            captions.extend(c.strip() for c in decoded)
        return captions
    finally:
        processor.tokenizer.padding_side = original_padding_side


def caption_dataset(
    data_path: str,
    image_dir: str,
    model,
    processor,
    key: str = "generated_caption_smolvlm",
    prompt: str = DEFAULT_CAPTION_PROMPT,
    batch_size: int = 8,
    max_new_tokens: int = 96,
    overwrite: bool = False,
    ids: Optional[Iterable[int]] = None,
) -> None:
    """Caption images for samples in ``data_path`` (JSON list) and write back.

    Idempotent: skips samples that already have ``key`` populated unless
    ``overwrite=True``. Skips samples whose image file is missing.
    """
    with open(data_path) as fh:
        data = json.load(fh)
    by_id = {s["id"]: s for s in data}
    id_filter = set(ids) if ids is not None else None

    targets = []
    missing_image = 0
    for s in data:
        if id_filter is not None and s["id"] not in id_filter:
            continue
        if not overwrite and s.get(key):
            continue
        img_path = os.path.join(image_dir, f"{s['id']}.png")
        if not os.path.exists(img_path):
            missing_image += 1
            continue
        targets.append(s)

    print(f"Captioning {len(targets)} samples → key '{key}' (skipping {missing_image} with missing images)")
    if not targets:
        return

    for start in tqdm(range(0, len(targets), batch_size), desc="caption"):
        chunk = targets[start : start + batch_size]
        imgs = [
            Image.open(os.path.join(image_dir, f"{s['id']}.png")).convert("RGB")
            for s in chunk
        ]
        caps = caption_images_batched(
            imgs,
            model,
            processor,
            prompt=prompt,
            batch_size=len(imgs),
            max_new_tokens=max_new_tokens,
        )
        for s, c in zip(chunk, caps):
            by_id[s["id"]][key] = c

    with open(data_path, "w") as fh:
        json.dump(data, fh, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SmolVLM2 captioner")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--key", default="generated_caption_smolvlm")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ids", nargs="*", type=int, default=None,
                        help="Optional subset of sample ids to caption.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, processor = load_smolvlm2(model_id=args.model_id, device=args.device)
    caption_dataset(
        data_path=args.data_path,
        image_dir=args.image_dir,
        model=model,
        processor=processor,
        key=args.key,
        prompt=args.prompt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        overwrite=args.overwrite,
        ids=args.ids,
    )


if __name__ == "__main__":
    main()
