import pytest
from backend.evidence.schemas import BoundingBox, UncertaintyReason, UncertaintySignal


def test_bounding_box_geometry():
    bbox = BoundingBox(x_min=10.0, y_min=20.0, x_max=50.0, y_max=80.0)
    assert bbox.width == 40.0
    assert bbox.height == 60.0
    assert bbox.area == 2400.0
    assert bbox.centroid == (30.0, 50.0)


def test_bounding_box_inverted_coordinates():
    bbox = BoundingBox(x_min=50.0, y_min=80.0, x_max=10.0, y_max=20.0)
    assert bbox.width == 0.0
    assert bbox.height == 0.0
    assert bbox.area == 0.0


def test_uncertainty_signal_serialization():
    signal = UncertaintySignal(
        reason=UncertaintyReason.BLUR,
        severity="HIGH",
        metric_name="laplacian_variance",
        metric_value=42.5,
        threshold=100.0,
        description="High blur detected",
    )
    dumped = signal.model_dump()
    assert dumped["reason"] == "BLUR"
    assert dumped["metric_value"] == 42.5
