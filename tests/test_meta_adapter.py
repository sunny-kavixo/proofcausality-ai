from backend.evidence.schemas import BoundingBox
from backend.providers.meta_adapter import MetaAdapter, redact_token, sanitize_error_detail
import numpy as np


def test_redact_token():
    assert redact_token(None) == "[NOT_SET]"
    assert redact_token("") == "[NOT_SET]"
    assert redact_token("meta-secret-key-12345") == "[REDACTED]"


def test_sanitize_error_detail():
    err = "Auth failed: Bearer super_secret_token_1234567890abc on gateway"
    sanitized = sanitize_error_detail(err)
    assert "super_secret" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized


def test_meta_adapter_fallback_mode():
    adapter = MetaAdapter(api_key="", enabled=False)
    diag = adapter.get_diagnostics()
    assert diag["configured"] is False
    assert diag["active_mode"] == "EvidenceCore Deterministic Fallback"

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = BoundingBox(x_min=10, y_min=10, x_max=40, y_max=40)
    mask = adapter.predict_mask(img, prompt_bbox=bbox)
    assert mask.shape == (100, 100)
    assert bool(mask[20, 20]) is True
