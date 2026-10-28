"""Detector-Independence Audit & Correlated Error Fusion (H04).

Addresses the failure mode where multiple vision models agree on an incorrect prediction
due to shared dataset bias or architectural correlation (e.g. common CNN/ViT priors).

Evaluates:
- Pairwise error correlation matrix between detectors.
- Comparison of Single-Model vs Majority Vote vs Confidence-Weighted vs Dependence-Aware Fusion.
- Calibration under distribution shift (e.g. synthetic lighting or heavy shadows).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np


class DetectorIndependenceAuditor:
    """Audits detector correlation and executes dependence-aware multi-model arbitration."""

    def __init__(self, correlation_discount: float = 0.5) -> None:
        self.correlation_discount = correlation_discount

    def compute_empirical_error_correlation(
        self,
        detector_a_errors: List[bool],
        detector_b_errors: List[bool],
    ) -> float:
        """Computes Pearson phi coefficient of binary error vectors."""
        if len(detector_a_errors) != len(detector_b_errors) or len(detector_a_errors) == 0:
            return 0.0

        a = np.array(detector_a_errors, dtype=float)
        b = np.array(detector_b_errors, dtype=float)

        std_a = np.std(a)
        std_b = np.std(b)

        if std_a == 0.0 or std_b == 0.0:
            return 0.0

        corr = float(np.corrcoef(a, b)[0, 1])
        return round(float(corr), 4)

    def arbitrate_multi_model_verdicts(
        self,
        predictions: Dict[str, bool],
        confidences: Dict[str, float],
        pairwise_correlations: Dict[Tuple[str, str], float],
    ) -> Dict[str, Any]:
        """Compares fusion strategies: Single Model, Majority, Confidence-Weighted, and Dependence-Aware."""
        models = list(predictions.keys())
        if not models:
            return {"verdict": False, "score": 0.0, "method": "EMPTY"}

        # 1. Single Model (First model)
        single_verdict = predictions[models[0]]
        single_score = confidences[models[0]]

        # 2. Majority Vote
        positive_votes = sum(1 for m in models if predictions[m])
        majority_verdict = positive_votes > (len(models) / 2.0)

        # 3. Standard Confidence-Weighted Fusion
        conf_sum = sum(confidences.values())
        if conf_sum > 0:
            weighted_pos = sum(confidences[m] for m in models if predictions[m]) / conf_sum
        else:
            weighted_pos = 0.5
        conf_verdict = weighted_pos >= 0.5

        # 4. Dependence-Aware Fusion:
        # Discount the effective weight of model j by its mean correlation with earlier models
        effective_weights = {}
        for i, m_i in enumerate(models):
            correlations = []
            for j, m_j in enumerate(models):
                if i != j:
                    pair = (m_i, m_j) if (m_i, m_j) in pairwise_correlations else (m_j, m_i)
                    correlations.append(pairwise_correlations.get(pair, 0.0))
            avg_corr = max(0.0, np.mean(correlations)) if correlations else 0.0
            # Weight discount: 1 / (1 + correlation_discount * avg_corr)
            effective_weights[m_i] = confidences[m_i] / (1.0 + self.correlation_discount * avg_corr)

        total_eff_weight = sum(effective_weights.values())
        if total_eff_weight > 0:
            dep_aware_score = sum(effective_weights[m] for m in models if predictions[m]) / total_eff_weight
        else:
            dep_aware_score = 0.5
        dep_aware_verdict = dep_aware_score >= 0.5

        return {
            "single_model": {"verdict": single_verdict, "confidence": round(single_score, 3)},
            "majority_vote": {"verdict": majority_verdict, "vote_ratio": round(positive_votes / len(models), 3)},
            "confidence_weighted": {"verdict": conf_verdict, "weighted_score": round(weighted_pos, 3)},
            "dependence_aware": {"verdict": dep_aware_verdict, "calibrated_score": round(dep_aware_score, 3)},
        }
