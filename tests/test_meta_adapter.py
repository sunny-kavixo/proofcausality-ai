import pytest
import httpx
import numpy as np

from backend.evidence.schemas import BoundingBox
from backend.providers.meta_adapter import (
    MetaAdapter,
    MetaAuthError,
    MetaRateLimitError,
    MetaProviderError,
    MetaShapeMismatchError,
    redact_token,
    sanitize_error_detail,
)


def test_redact_token():
    assert redact_token(None) == "[NOT_SET]"
    assert redact_token("") == "[NOT_SET]"
    assert "2345" in redact_token("meta-secret-key-12345")
    assert "REDACTED" in redact_token("meta-secret-key-12345")


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
    assert diag["provider_status"] == "BLOCKED_NO_CREDENTIALS"
    assert diag["meta_inference_status"] == "META_INFERENCE_BLOCKED"
    assert "META_API_KEY" in diag["meta_inference_explanation"]

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = BoundingBox(x_min=10, y_min=10, x_max=40, y_max=40)
    mask, meta = adapter.predict_mask(img, prompt_bbox=bbox)
    assert mask.shape == (100, 100)
    assert bool(mask[20, 20]) is True
    assert meta["source"] == "GEOMETRIC_FALLBACK"


def test_meta_adapter_mock_success_200():
    # Mock transport simulating Meta SAM 2.1 200 OK response
    expected_mask = np.zeros((50, 50), dtype=bool)
    expected_mask[10:30, 10:30] = True

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-meta-token-12345"
        return httpx.Response(200, json={"mask": expected_mask.tolist()})

    transport = httpx.MockTransport(handler)
    adapter = MetaAdapter(
        api_key="test-meta-token-12345",
        sam_endpoint="https://api.meta.mock/v1/sam",
        enabled=True,
        transport=transport,
    )

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask, meta = adapter.predict_mask(img)
    assert mask.shape == (50, 50)
    assert meta["source"] == "META_SAM_REMOTE"
    assert meta["http_status"] == 200
    assert np.array_equal(mask, expected_mask)


def test_meta_adapter_mock_auth_401():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized key"})

    transport = httpx.MockTransport(handler)
    adapter = MetaAdapter(
        api_key="invalid-key",
        sam_endpoint="https://api.meta.mock/v1/sam",
        enabled=True,
        transport=transport,
    )

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(MetaAuthError, match="authentication failed"):
        adapter.predict_mask(img)


def test_meta_adapter_mock_rate_limit_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "Rate limit exceeded"})

    transport = httpx.MockTransport(handler)
    adapter = MetaAdapter(
        api_key="valid-key",
        sam_endpoint="https://api.meta.mock/v1/sam",
        enabled=True,
        transport=transport,
    )

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(MetaRateLimitError, match="rate limit exceeded"):
        adapter.predict_mask(img)


def test_meta_adapter_mock_shape_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        # Returns 10x10 mask when input is 50x50
        return httpx.Response(200, json={"mask": np.zeros((10, 10), dtype=bool).tolist()})

    transport = httpx.MockTransport(handler)
    adapter = MetaAdapter(
        api_key="valid-key",
        sam_endpoint="https://api.meta.mock/v1/sam",
        enabled=True,
        transport=transport,
    )

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(MetaShapeMismatchError, match="does not match"):
        adapter.predict_mask(img)


def test_meta_adapter_mock_not_found_404():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "Model or endpoint not found"})

    transport = httpx.MockTransport(handler)
    adapter = MetaAdapter(
        api_key="valid-key",
        sam_endpoint="https://api.meta.mock/v1/sam",
        enabled=True,
        transport=transport,
    )

    img = np.zeros((50, 50, 3), dtype=np.uint8)
    from backend.providers.meta_adapter import MetaNotFoundError
    with pytest.raises(MetaNotFoundError, match="not found"):
        adapter.predict_mask(img)


def test_meta_adapter_llama_vision_offline_fallback():
    adapter = MetaAdapter(enabled=False)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = adapter.audit_scene_context_with_llama(img)
    assert res["status"] == "OFFLINE_FALLBACK"
    assert "BLOCKED" in res["response_text"]
    assert res["latency_ms"] >= 0.0
