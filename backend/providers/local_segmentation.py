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
EXPECTED_CHECKPOINT_SHA256 = "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80"



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

    def __init__(
        self,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
        expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.local_files_only = local_files_only
        self._processor: Any = None
        self._model: Any = None
        self._torch: Any = None
        self.device = "unloaded"
        self.checkpoint_path: Path | None = None
        self._checkpoint_sha256: str | None = None
        self._checkpoint_size_bytes: int = 0
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

                if self.revision != MODEL_REVISION:
                    raise ModelUnavailableError(
                        f"Unsupported model revision: {self.revision}. Expected immutable pinned revision {MODEL_REVISION}."
                    )

                self._processor = AutoImageProcessor.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    local_files_only=self.local_files_only,
                )
                self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    local_files_only=self.local_files_only,
                    use_safetensors=True,
                )

                # Expected architecture verification
                expected_arch = "Mask2FormerForUniversalSegmentation"
                model_archs = getattr(self._model.config, "architectures", [])
                if not isinstance(self._model, Mask2FormerForUniversalSegmentation) and expected_arch not in model_archs:
                    raise ModelUnavailableError(
                        f"Unexpected model architecture: {model_archs}; expected {expected_arch}"
                    )

                # Model output class mapping verification
                id2label = getattr(self._model.config, "id2label", None)
                if not id2label or not isinstance(id2label, dict) or len(id2label) == 0:
                    raise ModelUnavailableError("Model config is missing required id2label class mapping.")

                device = "cpu"
                if torch.backends.mps.is_available():
                    try:
                        test_tensor = torch.zeros((1,), device="mps")
                        del test_tensor
                        device = "mps"
                    except Exception:
                        device = "cpu"
                self.device = device
                self._model.to(self.device).eval()
                self._torch = torch

                # Pinned safetensors checkpoint resolution
                candidate = Path(self._model.name_or_path)
                if candidate.exists() and (candidate / "model.safetensors").is_file():
                    self.checkpoint_path = candidate / "model.safetensors"
                else:
                    from huggingface_hub import hf_hub_download

                    try:
                        self.checkpoint_path = Path(
                            hf_hub_download(
                                self.model_id,
                                "model.safetensors",
                                revision=self.revision,
                                local_files_only=self.local_files_only,
                            )
                        )
                    except Exception as exc:
                        raise ModelUnavailableError(
                            f"Missing or inaccessible safetensors checkpoint for {self.model_id} (revision {self.revision}): {exc}"
                        ) from exc

                if not self.checkpoint_path or not self.checkpoint_path.is_file():
                    raise ModelUnavailableError(f"Cached safetensors checkpoint not found at {self.checkpoint_path}")

                self._checkpoint_size_bytes = self.checkpoint_path.stat().st_size
                actual_sha256 = sha256_bytes(self.checkpoint_path.read_bytes())
                self._checkpoint_sha256 = actual_sha256

                # Cryptographic checkpoint identity validation
                if actual_sha256.lower() != self.expected_checkpoint_sha256.lower():
                    raise ModelUnavailableError(
                        f"Checkpoint SHA-256 verification failed: expected {self.expected_checkpoint_sha256}, got {actual_sha256}"
                    )
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(f"Local segmentation model unavailable: {type(exc).__name__}: {exc}") from exc

    def predict_from_rgb(
        self,
        rgb_arr: np.ndarray,
        threshold: float = 0.5,
        raw_bytes: bytes | None = None,
    ) -> tuple[list[InstancePrediction], InferenceExecution]:
        """Perform instance segmentation directly from an RGB numpy array."""
        self.load()
        height, width, _ = rgb_arr.shape
        image = Image.fromarray(rgb_arr)

        with self._inference_lock:
            try:
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
                input_hash = sha256_bytes(raw_bytes) if raw_bytes else sha256_bytes(rgb_arr.tobytes())
                execution = InferenceExecution(
                    provider="Hugging Face Transformers / local PyTorch",
                    model_identifier=self.model_id,
                    model_revision=self.revision,
                    inference_mode="UNPROMPTED_INSTANCE_SEGMENTATION",
                    device=self.device,
                    input_sha256=input_hash,
                    input_width=width,
                    input_height=height,
                    prompt=None,
                    checkpoint_sha256=self._checkpoint_sha256 or "UNKNOWN",
                    checkpoint_size_bytes=self._checkpoint_size_bytes,
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
            except (ModelUnavailableError, ModelOutputError):
                raise
            except Exception as exc:
                raise ModelUnavailableError(f"Model inference failed: {type(exc).__name__}: {exc}") from exc
            finally:
                if self.device == "mps" and self._torch is not None:
                    try:
                        if hasattr(self._torch, "mps") and hasattr(self._torch.mps, "empty_cache"):
                            self._torch.mps.empty_cache()
                    except Exception:
                        pass

    def predict(self, image_path: str | Path, threshold: float = 0.5) -> tuple[list[InstancePrediction], InferenceExecution]:
        path = Path(image_path)
        image_bytes = path.read_bytes()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            rgb_arr = np.array(image, dtype=np.uint8)
        return self.predict_from_rgb(rgb_arr, threshold=threshold, raw_bytes=image_bytes)


import threading

_PROVIDER_LOCK = threading.Lock()
_CACHED_PROVIDER: LocalMask2FormerProvider | None = None


def get_local_mask2former_provider() -> LocalMask2FormerProvider:
    """Thread-safe singleton accessor for LocalMask2FormerProvider."""
    global _CACHED_PROVIDER
    with _PROVIDER_LOCK:
        if _CACHED_PROVIDER is None:
            _CACHED_PROVIDER = LocalMask2FormerProvider()
        return _CACHED_PROVIDER


def unload_local_mask2former_provider() -> None:
    """Unload cached model and release memory."""
    global _CACHED_PROVIDER
    with _PROVIDER_LOCK:
        if _CACHED_PROVIDER is not None:
            if _CACHED_PROVIDER._torch is not None and _CACHED_PROVIDER.device == "mps":
                if hasattr(_CACHED_PROVIDER._torch, "mps") and hasattr(_CACHED_PROVIDER._torch.mps, "empty_cache"):
                    _CACHED_PROVIDER._torch.mps.empty_cache()
            _CACHED_PROVIDER = None


