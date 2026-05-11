"""Utility helpers for feature extraction and preprocessing of datasets."""

import os

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    SiglipImageProcessor,
)


SIGLIP_DEFAULT_MAX_TEXT_LENGTH = 64


def _unwrap_features(out):
    """Pull a (B, D) feature tensor out of a SigLIP forward result.

    Older transformers returned a tensor directly; newer ones may return a
    ``BaseModelOutputWithPooling`` (or similar) — handle both.
    """
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if val is not None:
                return val
    if hasattr(out, "last_hidden_state"):
        # Mean-pool fallback for sequence outputs.
        return out.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unexpected SigLIP output type: {type(out).__name__}")


def build_siglip2(model_id: str = "google/siglip2-base-patch16-224", device: str = "cuda"):
    """Load a SigLIP2 model plus tokenizer and image processor for the requested device."""
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    image_processor = SiglipImageProcessor.from_pretrained(model_id)
    return model, tokenizer, image_processor


def _tokenize_siglip_texts(tokenizer, texts: List[str]):
    """Tokenize text for SigLIP with fixed-length padding; fall back when model_max_length is huge."""
    tokenizer_kwargs = {
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }
    max_len = getattr(tokenizer, "model_max_length", SIGLIP_DEFAULT_MAX_TEXT_LENGTH)
    if not isinstance(max_len, int) or max_len <= 0 or max_len > 512:
        max_len = SIGLIP_DEFAULT_MAX_TEXT_LENGTH
    tokenizer_kwargs["max_length"] = max_len
    return tokenizer(texts, **tokenizer_kwargs)


def build_qwen3_vl(
    model_id: str = "Qwen/Qwen3-VL-32B-Instruct",
    device: Optional[str] = None,
):
    """Load a Qwen3 vision-language model for open-source captioning."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor, device


def batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    """Move a batch dictionary returned by a processor onto the target device."""
    moved: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def extract_image_features_siglip2(
    pil_images: List[Image.Image],
    model,
    image_processor,
    batch_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute normalized SigLIP2 image embeddings for a list of PIL images."""
    if not pil_images:
        projection_dim = getattr(model.config, "projection_dim", None)
        if projection_dim is None:
            projection_dim = getattr(
                model.config.vision_config,
                "projection_size",
                getattr(model.config.vision_config, "hidden_size"),
            )
        return torch.empty((0, projection_dim), dtype=torch.float32)

    feats: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(pil_images), batch_size):
        batch_imgs = [img.convert("RGB") for img in pil_images[start : start + batch_size]]
        inputs = image_processor(images=batch_imgs, return_tensors="pt")
        inputs = batch_to_device(inputs, device)
        with torch.no_grad():
            image_features = _unwrap_features(model.get_image_features(**inputs))
            image_features = F.normalize(image_features, dim=-1)
        feats.append(image_features.cpu().float())
    return torch.cat(feats, dim=0)


def extract_text_features_siglip2(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    """Compute normalized SigLIP2 text embeddings for a list of texts."""
    if not texts:
        projection_dim = getattr(model.config, "projection_dim", None)
        if projection_dim is None:
            projection_dim = getattr(
                model.config.text_config,
                "projection_size",
                getattr(model.config.text_config, "hidden_size"),
            )
        return torch.empty((0, projection_dim), dtype=torch.float32)

    feats: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch_texts = [t.lower() for t in texts[start : start + batch_size]]
        inputs = _tokenize_siglip_texts(tokenizer, batch_texts)
        text_inputs = batch_to_device(inputs, device)
        with torch.no_grad():
            text_features = _unwrap_features(model.get_text_features(**text_inputs))
            text_features = F.normalize(text_features, dim=-1)
        feats.append(text_features.cpu().float())
    return torch.cat(feats, dim=0)


def stratified_split_indices(
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Return stratified indices for train/val/test splits."""
    rs = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    idx_train: List[int] = []
    idx_val: List[int] = []
    idx_test: List[int] = []
    for label in classes:
        idx = np.where(labels == label)[0]
        rs.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        idx_train.extend(idx[:n_train])
        idx_val.extend(idx[n_train : n_train + n_val])
        idx_test.extend(idx[n_train + n_val :])
    rs.shuffle(idx_train)
    rs.shuffle(idx_val)
    rs.shuffle(idx_test)
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def pca_fit_on_train_and_transform(
    x_all: torch.Tensor,
    idx_train: np.ndarray,
    target_dim: int,
) -> Dict[str, torch.Tensor]:
    """Fit low-rank PCA on the train subset and project all rows to target_dim."""
    x_all = x_all.float().cpu()
    x_train = x_all[idx_train]
    mu = x_train.mean(dim=0, keepdim=True)
    x_centered = x_train - mu
    # torch.pca_lowrank requires q <= min(M, N).
    n_rows, n_cols = x_centered.shape
    q = max(1, min(target_dim, n_cols - 1 if n_cols > 1 else 1, n_rows))
    _, singular_vals, v_matrix = torch.pca_lowrank(x_centered, q=q)
    components = v_matrix[:, :q]
    singular_vals_k = singular_vals[:q]
    explained_variance = (singular_vals_k ** 2) / max(1, (x_centered.shape[0] - 1))
    if explained_variance.sum() > 0:
        explained_variance_ratio = explained_variance / explained_variance.sum()
    else:
        explained_variance_ratio = torch.zeros_like(explained_variance)
    x_all_reduced = (x_all - mu) @ components
    return {
        "reduced": x_all_reduced,
        "mean": mu.squeeze(0),
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "effective_dim": int(components.shape[1]),
    }


def save_feature_tensors(features: Dict[str, torch.Tensor], out_path: str) -> str:
    """Optionally persist feature tensors to a local .pt file."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(features, out_path)
    return out_path
