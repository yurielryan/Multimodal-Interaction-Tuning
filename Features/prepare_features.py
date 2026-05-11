"""Reusable SigLIP2 feature extraction for multimodal interaction estimation.

This module expects batches from a dataloader and returns feature tensors.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from Features.utils_features import (
    build_siglip2,
    extract_image_features_siglip2,
    extract_text_features_siglip2,
    pca_fit_on_train_and_transform,
)

SIGLIP2_MODEL_ID = "google/siglip2-giant-opt-patch16-384"
PCA_TARGET_DIM = 512


def load_siglip2_extractor(
    model_id: str = SIGLIP2_MODEL_ID,
    device: Optional[str] = None,
):
    """Load SigLIP2 model stack and return (model, tokenizer, image_processor, device_str)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, image_processor = build_siglip2(model_id=model_id, device=device) # build_siglip2 from ./Features/utils_features.py
    return model, tokenizer, image_processor, str(model.device)

def extract_features_from_batch(
    batch: Any,
    model,
    tokenizer,
    image_processor,
    device: str,
    pca_512: bool = True,
    image_batch_size: int = 32,
    text_batch_size: int = 128, # different sizes because text is usually faster. sample alignment should be maintained by the dataloader, so we can process them separately.
) -> Dict[str, Any]:
    """Extract SigLIP2 features from a dataloader batch.

    Supports:
    - dict batch: {'modal_1': ..., 'modal_2': ...}
    - tuple/list batch: (modal_1, modal_2, ...)
    """
    images: List[Image.Image] = []
    texts: List[str] = []

    modal_1_batch, modal_2_batch = _parse_batch(batch)
    modal_1_items = _to_list(modal_1_batch)
    modal_2_items = _to_list(modal_2_batch)

    if len(modal_1_items) != len(modal_2_items): # check for missing/additional samples
        raise ValueError("Image and text batch sizes must match.")

    for image_item, text_item in zip(modal_1_items, modal_2_items):
        images.append(_extract_image(image_item))
        texts.append(_extract_text(text_item))

    modal_1_features = extract_image_features_siglip2(
        images,
        model,
        image_processor,
        batch_size=image_batch_size,
        device=device,
    )
    modal_2_features = extract_text_features_siglip2(
        texts,
        model,
        tokenizer,
        batch_size=text_batch_size,
        device=device,
    )

    if pca_512:
        if modal_1_features.shape[0] < 2 or modal_2_features.shape[0] < 2:
            raise ValueError("PCA-512 requires at least 2 samples in a batch.")
        idx_all = np.arange(modal_1_features.shape[0])
        modal_1_features = pca_fit_on_train_and_transform(modal_1_features, idx_all, PCA_TARGET_DIM)["reduced"]
        modal_2_features = pca_fit_on_train_and_transform(modal_2_features, idx_all, PCA_TARGET_DIM)["reduced"]

    return {
        "modal_1_features": modal_1_features, # shape: [N x D]
        "modal_2_features": modal_2_features, # shape: [N x D] where N is batch size, and D would be 512 if PCA is performed.
    }


def _parse_batch(batch: Any) -> Tuple[Sequence[Any], Sequence[Any]]:
    """Parse common dataloader batch formats into (images, texts).

    Supported formats:
    - dict with keys modal_1 (image) and modal_2 (text)
    - tuple/list where first element is image batch and second is text batch
    """
    if isinstance(batch, dict):
        if "modal_1" not in batch or "modal_2" not in batch:
            raise ValueError("Dict batch must contain keys 'modal_1' and 'modal_2'.")
        return batch["modal_1"], batch["modal_2"] # return only a tuple of (image, text) where both image and text are of size (N,)

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise ValueError("Unsupported batch format. Use dict(modal_1, modal_2) or tuple/list(batch_images, batch_texts, ...).")


def _to_list(values: Any) -> List[Any]: # convert to list
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if torch.is_tensor(values):
        return list(values)
    return list(values)


def _extract_image(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, dict): # to handle dictionaries
        if "modal_1" in item and isinstance(item["modal_1"], Image.Image):
            return item["modal_1"]
        if "image" in item and isinstance(item["image"], Image.Image):
            return item["image"]
    raise ValueError("Each image item must be a PIL image (or dict containing one).")


def _extract_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if "modal_2" in item and item["modal_2"] is not None:
            return str(item["modal_2"])
        if "text" in item and item["text"] is not None:
            return str(item["text"])
    return str(item)
