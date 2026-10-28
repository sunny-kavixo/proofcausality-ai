from backend.evidence.recapture import plan_recapture
from backend.evidence.schemas import UncertaintyReason, UncertaintySignal


def test_plan_recapture_blur():
    signal = UncertaintySignal(
        reason=UncertaintyReason.BLUR,
        severity="HIGH",
        metric_name="laplacian_variance",
        metric_value=25.0,
        threshold=100.0,
        description="High blur",
    )
    plan = plan_recapture([signal])
    assert plan is not None
    assert plan.reason == UncertaintyReason.BLUR
    assert "Hold the camera steady" in plan.directive


def test_plan_recapture_priority():
    sig_blur = UncertaintySignal(
        reason=UncertaintyReason.BLUR,
        severity="MEDIUM",
        metric_name="laplacian_variance",
        metric_value=80.0,
        threshold=100.0,
        description="Blur",
    )
    sig_occ = UncertaintySignal(
        reason=UncertaintyReason.OCCLUSION,
        severity="HIGH",
        metric_name="area_ratio",
        metric_value=0.2,
        threshold=0.65,
        description="Occlusion",
    )
    # Occlusion has higher priority than blur
    plan = plan_recapture([sig_blur, sig_occ])
    assert plan is not None
    assert plan.reason == UncertaintyReason.OCCLUSION
    assert "Keep the object visible" in plan.directive
