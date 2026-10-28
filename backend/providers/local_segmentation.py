"""Fail-closed local instance-segmentation inference with execution provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


MODEL_ID = "facebook/mask2former-swin-tiny-coco-instance"
MODEL_REVISION = "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"


class ModelUnavailableError(RuntimeError):
    """Raised when genuine inference cannot be executed locally."""


class ModelOutputError(RuntimeError):
    """Raised when a model response is empty, malformed, or dimensionally invalid."""


@dataclass(frozen=True)
class InstancePrediction:
    instance_id: int
    label_id: int
    label: str
    score: float
    mask: np.ndarray


@dataclass(frozen=True)
class InferenceExecution:
    provider: str
    model_identifier: str
    model_revision: str
    inference_mode: str
    device: str
    input_sha256: str
    input_width: int
    input_height: int
    prompt: None
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    output_sha256: str
    latency_ms: float
    execution_status: str
    runtime: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_binary_mask_rle(mask: np.ndarray) -> list[int]:
    """Encode a boolean mask as deterministic row-major run lengths."""
    if mask.ndim != 2:
        raise ModelOutputError(f"Mask must be 2-D, got shape {mask.shape}")
    flat = mask.astype(np.uint8, copy=False).ravel(order="C")
    runs: list[int] = []
    previous = 0
    count = 0
    for value in flat:
        bit = int(value)
        if bit == previous:
            count += 1
        else:
            runs.append(count)
            count = 1
            previous = bit
    runs.append(count)
    return runs


def decode_binary_mask_rle(runs: list[int], height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0 or not runs or any((not isinstance(n, int) or n < 0) for n in runs):
        raise ModelOutputError("Invalid binary-mask RLE")
    values: list[np.ndarray] = []
    bit = False
    for count in runs:
        if count:
            values.append(np.full(count, bit, dtype=bool))
        bit = not bit
    flat = np.concatenate(values) if values else np.empty(0, dtype=bool)
    if flat.size != height * width:
        raise ModelOutputError(f"RLE decodes to {flat.size} pixels; expected {height * width}")
    return flat.reshape((height, width))


class LocalMask2FormerProvider:
    """Runs unprompted COCO instance segmentation on original image pixels.

    The prediction API intentionally accepts no annotation, reference mask, or box so
    reference-mask leakage is structurally excluded from prediction generation.
    """

    def __init__(self, model_id: str = MODEL_ID, local_files_only: bool = True) -> None:
        self.model_id = model_id
        self.local_files_only = local_files_only
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self.device = "unloaded"
        self.checkpoint_path: Path | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

            self._processor = AutoImageProcessor.from_pretrained(
                self.model_id, local_files_only=self.local_files_only
            )
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.model_id, local_files_only=self.local_files_only, use_safetensors=True
            )
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model.to(self.device).eval()
            self._torch = torch
            candidate = Path(self._model.name_or_path)
            if candidate.exists():
                self.checkpoint_path = candidate / "model.safetensors"
            else:
                from huggingface_hub import hf_hub_download

                self.checkpoint_path = Path(
                    hf_hub_download(self.model_id, "model.safetensors", local_files_only=True)
                )
            if not self.checkpoint_path.is_file():
                raise ModelUnavailableError("Cached safetensors checkpoint was not found")
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(f"Local segmentation model unavailable: {type(exc).__name__}: {exc}") from exc

    def predict(self, image_path: str | Path, threshold: float = 0.5) -> tuple[list[InstancePrediction], InferenceExecution]:
        self.load()
        path = Path(image_path)
        image_bytes = path.read_bytes()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        started = time.perf_counter()
        with self._torch.inference_mode():
            raw = self._model(**inputs)
        if self.device == "mps":
            self._torch.mps.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        processed = self._processor.post_process_instance_segmentation(
            raw, target_sizes=[(height, width)], threshold=threshold
        )
        if not isinstance(processed, list) or len(processed) != 1:
            raise ModelOutputError("Unsupported instance-segmentation output container")
        result = processed[0]
        segmentation = result.get("segmentation")
        segments_info = result.get("segments_info")
        if segmentation is None or tuple(segmentation.shape) != (height, width):
            shape = None if segmentation is None else tuple(segmentation.shape)
            raise ModelOutputError(f"Prediction dimension mismatch: {shape} != {(height, width)}")
        if not isinstance(segments_info, list):
            raise ModelOutputError("Missing segments_info in model output")
        segmentation_np = segmentation.detach().cpu().numpy()
        predictions: list[InstancePrediction] = []
        serializable: list[dict[str, Any]] = []
        for segment in segments_info:
            instance_id = int(segment["id"])
            label_id = int(segment["label_id"])
            mask = segmentation_np == instance_id
            if mask.shape != (height, width) or not np.any(mask):
                continue
            prediction = InstancePrediction(
                instance_id=instance_id,
                label_id=label_id,
                label=str(self._model.config.id2label.get(label_id, label_id)),
                score=float(segment["score"]),
                mask=mask,
            )
            predictions.append(prediction)
            serializable.append({
                "instance_id": instance_id,
                "label_id": label_id,
                "label": prediction.label,
                "score": prediction.score,
                "rle": encode_binary_mask_rle(mask),
            })
        output_bytes = json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode()
        checkpoint_bytes = self.checkpoint_path.read_bytes()
        execution = InferenceExecution(
            provider="Hugging Face Transformers / local PyTorch",
            model_identifier=self.model_id,
            model_revision=MODEL_REVISION,
            inference_mode="UNPROMPTED_INSTANCE_SEGMENTATION",
            device=self.device,
            input_sha256=sha256_bytes(image_bytes),
            input_width=width,
            input_height=height,
            prompt=None,
            checkpoint_sha256=sha256_bytes(checkpoint_bytes),
            checkpoint_size_bytes=len(checkpoint_bytes),
            output_sha256=sha256_bytes(output_bytes),
            latency_ms=round(latency_ms, 3),
            execution_status="SUCCEEDED",
            runtime={
                "python": platform.python_version(),
                "torch": str(self._torch.__version__),
                "platform": platform.platform(),
            },
        )
        return predictions, execution

