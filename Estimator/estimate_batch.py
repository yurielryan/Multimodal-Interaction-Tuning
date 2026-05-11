"""
Estimate multimodal interactions for a batch of samples.

Inputs (batch) from dataloader:
- modal_1: images from the dataset provided by dataloader (PIL format)
- modal_2: text from the dataset provided by dataloader (string format)

Outputs a tuple of MI estimates for that batch of samples, which can be saved to a file for later analysis.
- [N, 4] tensor of MI estimates for that batch of samples, where N is the batch size and the 4 columns correspond to the 4 terms in the MI estimator [R, U1, U2, S] for each sample in the batch.

"""
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from Features.prepare_features import extract_features_from_batch
from mi_estimator import RUS_adjustment

def estimate_rus_batch(
	batch: Any,
	discriminator: Sequence[torch.nn.Module],
	entropy_estimator: Sequence[torch.nn.Module],
	feature_model,
	feature_tokenizer,
	feature_image_processor,
	device: str,
	pca_512: bool = True,
	image_batch_size: int = 32,
	text_batch_size: int = 128,
	apply_adjustment: bool = True,
	labels: Optional[torch.Tensor] = None,
	class_priors: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
	"""Main entrypoint: extract features from batch, then estimate pointwise R/U1/U2/S.

	Args:
		batch: Dataloader batch containing image/text pairs (+ labels).
		discriminator: Loaded classifier estimators [x1, x2, joint].
		entropy_estimator: Loaded entropy estimators [x1, x2].
		feature_model/tokenizer/image_processor: Loaded SigLIP2 stack.
		device: Target device (e.g. "cuda" or "cpu").
	"""
	features = extract_features_from_batch(
		batch=batch,
		model=feature_model,
		tokenizer=feature_tokenizer,
		image_processor=feature_image_processor,
		device=device,
		pca_512=pca_512,
		image_batch_size=image_batch_size,
		text_batch_size=text_batch_size,
	)

	modal_1_features = features["modal_1_features"].to(device) # image features: [N, D]
	modal_2_features = features["modal_2_features"].to(device) # text features: [N, D]
	labels_tensor = labels.to(device).long() if labels is not None else _labels_from_batch(batch, device) 

	estimates = estimate_rus_from_features(
		modal_1_features=modal_1_features,
		modal_2_features=modal_2_features,
		labels=labels_tensor,
		discriminator=discriminator,
		entropy_estimator=entropy_estimator,
		apply_adjustment=apply_adjustment,
		class_priors=class_priors,
	)

	estimates["modal_1_features"] = modal_1_features
	estimates["modal_2_features"] = modal_2_features
	estimates["labels"] = labels_tensor
	return estimates

def estimate_rus_from_features(
	modal_1_features: torch.Tensor,
	modal_2_features: torch.Tensor,
	labels: torch.Tensor,
	discriminator: Sequence[torch.nn.Module],
	entropy_estimator: Sequence[torch.nn.Module],
	apply_adjustment: bool = True,
	class_priors: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
	"""Estimate pointwise R/U1/U2/S directly from prepared feature tensors."""
	if len(discriminator) != 3:
		raise ValueError("discriminator must be a sequence of 3 models: [x1, x2, joint].")
	if len(entropy_estimator) != 2:
		raise ValueError("entropy_estimator must be a sequence of 2 models: [x1, x2].")

	for model in discriminator:
		model.eval()
	for model in entropy_estimator:
		model.eval()

	with torch.no_grad():
		i_x1y = _mutual_info_from_classifier(modal_1_features, labels, discriminator[0], class_priors=class_priors) # I(X1;Y)=log p(y|x1)-log p(y), using non-uniform class prior p(y).
		i_x2y = _mutual_info_from_classifier(modal_2_features, labels, discriminator[1], class_priors=class_priors) # I(X2;Y)=log p(y|x2)-log p(y), same prior correction.
		joint = torch.cat([modal_1_features, modal_2_features], dim=1) # Build joint representation [X1, X2] for joint MI term.
		i_x1x2y = _mutual_info_from_classifier(joint, labels, discriminator[2], class_priors=class_priors) # I((X1,X2);Y)=log p(y|x1,x2)-log p(y).

		h_x1 = _ensure_vector(entropy_estimator[0](modal_1_features), modal_1_features.shape[0]) # Per-sample entropy proxy H(X1).
		h_x2 = _ensure_vector(entropy_estimator[1](modal_2_features), modal_2_features.shape[0]) # Per-sample entropy proxy H(X2).

		r_plus = torch.minimum(h_x1, h_x2) # Upper shared-information cap from the smaller marginal entropy.
		r_minus = torch.minimum(h_x1 - i_x1y, h_x2 - i_x2y) # Shared "uninformative" overlap after removing label info.
		r = r_plus - r_minus # Redundant information term R.
		u_1 = i_x1y - r # Unique contribution from modality 1: U1 = I(X1;Y) - R.
		u_2 = i_x2y - r # Unique contribution from modality 2: U2 = I(X2;Y) - R.
		s = i_x1x2y - r - u_1 - u_2 # Synergy is the residual in joint MI after subtracting R, U1, U2.

		if apply_adjustment:
			r, u_1, u_2, s = RUS_adjustment([r, u_1, u_2, s])

		rus_pointwise = torch.stack([r, u_1, u_2, s], dim=1)
		rus_mean = rus_pointwise.mean(dim=0)

	return {
		"rus_pointwise": rus_pointwise,
		"rus_mean": rus_mean,
		"r": r,
		"u1": u_1,
		"u2": u_2,
		"s": s,
	}

def _labels_from_batch(batch: Any, device: str) -> torch.Tensor:
	"""Extract labels from common dataloader batch formats."""
	labels = None
	if isinstance(batch, dict):
		for key in ("labels", "label", "targets", "target", "y"):
			if key in batch:
				labels = batch[key]
				break
	elif isinstance(batch, (tuple, list)) and len(batch) >= 3:
		labels = batch[2]

	if labels is None:
		raise ValueError(
			"Labels are required to estimate R/U/S terms. "
			"Provide labels in batch['labels'] (or label/targets/target/y) or as the 3rd tuple element."
		)

	if torch.is_tensor(labels):
		return labels.to(device).long()
	return torch.as_tensor(labels, dtype=torch.long, device=device)


def _ensure_vector(loss_values: torch.Tensor, target_len: int) -> torch.Tensor:
	"""Normalize model output to per-sample shape [N]."""
	if loss_values.dim() == 0:
		return loss_values.repeat(target_len)
	if loss_values.dim() > 1:
		return loss_values.view(target_len, -1).mean(dim=1)
	return loss_values


def _mutual_info_from_classifier(
	input_data: torch.Tensor,
	labels: torch.Tensor,
	model,
	class_priors: Optional[torch.Tensor] = None,
) -> torch.Tensor:
	"""Per-sample MI estimate: log p(y|x) - log p(y)."""
	logits = model(input_data)
	rows = torch.arange(input_data.shape[0], device=input_data.device)
	n_classes = int(logits.shape[1])
	if class_priors is None:
		counts = torch.bincount(labels.long(), minlength=n_classes).to(dtype=torch.float32, device=input_data.device)
		class_priors = counts / counts.sum().clamp_min(1.0) # Fallback: empirical p(y) from the current batch labels.
	else:
		class_priors = torch.as_tensor(class_priors, dtype=torch.float32, device=input_data.device)
		if class_priors.numel() != n_classes:
			raise ValueError(f"class_priors length ({class_priors.numel()}) must match n_classes ({n_classes}).")
		class_priors = class_priors / class_priors.sum().clamp_min(1e-12)

	log_class_priors = class_priors.clamp_min(1e-12).log()
	return torch.nn.functional.log_softmax(logits, dim=1)[rows, labels] - log_class_priors[labels] # i(x;y)=log p(y|x)-log p(y), computed in nats.


