"""Regression tests for model revision and checkpoint reproducibility (Issue 2).

Verifies:
- Pinned immutable model revision (22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6) accepted.
- Incorrect revision rejected fail-closed with ModelUnavailableError.
- Incorrect checkpoint SHA-256 rejected fail-closed with ModelUnavailableError.
- Missing checkpoint reported clearly.
- Model outputs remain independent of reference annotations (structural exclusion).
- Model execution records preserve accurate model revision, checkpoint hash, device, and runtime.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import pytest
import numpy as np

from backend.providers.local_segmentation import (
    MODEL_ID,
    MODEL_REVISION,
    EXPECTED_CHECKPOINT_SHA256,
    LocalMask2FormerProvider,
    ModelUnavailableError,
)

try:
    import torch
    import transformers
    _p = LocalMask2FormerProvider()
    _p.load()
    _MODEL_AVAILABLE = True
except Exception:
    _MODEL_AVAILABLE = False

requires_local_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="PyTorch, Transformers, or local Mask2Former checkpoint not available in test environment",
)


@requires_local_model
def test_correct_revision_accepted():
    """Correct pinned revision and checkpoint hash load successfully."""
    provider = LocalMask2FormerProvider(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        local_files_only=True,
    )
    provider.load()
    assert provider._model is not None
    assert provider._processor is not None
    assert provider._checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert provider.revision == MODEL_REVISION


@requires_local_model
def test_incorrect_revision_rejected():
    """An unpinned, altered, or incorrect revision must fail closed immediately."""
    provider = LocalMask2FormerProvider(
        model_id=MODEL_ID,
        revision="ffffffffffffffffffffffffffffffffffffffff",  # Wrong revision
        local_files_only=True,
    )
    with pytest.raises(ModelUnavailableError, match="Unsupported model revision"):
        provider.load()


@requires_local_model
def test_incorrect_checkpoint_hash_rejected():
    """An altered expected checkpoint hash must be rejected even if revision string matches."""
    provider = LocalMask2FormerProvider(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        expected_checkpoint_sha256="0" * 64,  # Forged/tampered expected hash
        local_files_only=True,
    )
    with pytest.raises(ModelUnavailableError, match="Checkpoint SHA-256 verification failed"):
        provider.load()


def test_missing_checkpoint_reported_clearly(tmp_path: Path):
    """A non-existent checkpoint path or repo must fail with a clear diagnostic."""
    provider = LocalMask2FormerProvider(
        model_id="nonexistent-org/nonexistent-model",
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    with pytest.raises(ModelUnavailableError, match="Local segmentation model unavailable"):
        provider.load()


def test_model_outputs_remain_independent_of_reference_annotations():
    """Prediction interfaces structurally exclude reference annotations, masks, or bboxes."""
    sig_predict = inspect.signature(LocalMask2FormerProvider.predict)
    assert list(sig_predict.parameters.keys()) == ["self", "image_path", "threshold"]

    sig_predict_rgb = inspect.signature(LocalMask2FormerProvider.predict_from_rgb)
    params = list(sig_predict_rgb.parameters.keys())
    assert "annotation" not in params
    assert "reference_mask" not in params
    assert "ground_truth" not in params
    assert "bbox" not in params


@requires_local_model
def test_model_execution_records_preserve_accurate_identity():
    """Inference execution metadata faithfully records pinned revision and verified checkpoint hash."""
    provider = LocalMask2FormerProvider()
    dummy_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    preds, execution = provider.predict_from_rgb(dummy_rgb, threshold=0.5)

    assert execution.model_identifier == MODEL_ID
    assert execution.model_revision == MODEL_REVISION
    assert execution.checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256
    assert execution.execution_status == "SUCCEEDED"
    assert execution.device in ("mps", "cpu")
    assert execution.input_width == 100
    assert execution.input_height == 100
