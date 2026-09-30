# src/decision_layer.py

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class DecisionAction(str, Enum):
    RETAKE_IMAGE = "retake_image"
    URGENT_SPECIALIST_REVIEW = "urgent_specialist_review"
    SPECIALIST_REVIEW = "specialist_review"
    PRACTITIONER_REVIEW = "practitioner_review"
    ROUTINE_FOLLOWUP = "routine_followup"
    LOW_RISK_MONITOR = "low_risk_monitor"


RISK_RANK = {
    RiskLevel.UNKNOWN: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MODERATE: 2,
    RiskLevel.HIGH: 3,
}


ACTION_RANK = {
    DecisionAction.LOW_RISK_MONITOR: 1,
    DecisionAction.ROUTINE_FOLLOWUP: 2,
    DecisionAction.PRACTITIONER_REVIEW: 3,
    DecisionAction.SPECIALIST_REVIEW: 4,
    DecisionAction.URGENT_SPECIALIST_REVIEW: 5,
    DecisionAction.RETAKE_IMAGE: 6,
}


@dataclass
class ImageQualityAssessment:
    """
    Optional output from an image-quality module.

    status:
      - "pass": usable image
      - "warn": usable with caution
      - "fail": do not use model result; retake image

    score:
      Optional numeric quality score from 0.0 to 1.0.
    """

    status: str = "pass"
    score: Optional[float] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class GradCAMAssessment:
    """
    Optional assessment of Grad-CAM output.

    target_class_name:
      Class for which this Grad-CAM was generated.

    target_probability:
      Model probability for that class.

    focus_in_lesion_score:
      0.0 to 1.0 score estimating how much Grad-CAM focus is inside the lesion.
      This requires a lesion mask to be meaningful.

    artifact_focus:
      True if Grad-CAM appears to focus on artifact/background rather than lesion.

    heatmap_path:
      Path to saved raw Grad-CAM heatmap image.

    overlay_path:
      Path to saved Grad-CAM overlay image.
    """

    target_class_name: Optional[str] = None
    target_probability: Optional[float] = None
    target_layer: Optional[str] = None
    focus_in_lesion_score: Optional[float] = None
    artifact_focus: bool = False
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    selection_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_class_name": self.target_class_name,
            "target_probability": self.target_probability,
            "target_layer": self.target_layer,
            "focus_in_lesion_score": self.focus_in_lesion_score,
            "artifact_focus": self.artifact_focus,
            "heatmap_path": self.heatmap_path,
            "overlay_path": self.overlay_path,
            "selection_reasons": self.selection_reasons,
            "notes": self.notes,
        }


@dataclass
class DecisionConfig:
    """
    SME-owned decision-layer configuration.

    These are placeholder thresholds for engineering/testing.
    They should be adjusted after validation with domain experts.
    """

    class_risk_map: Dict[str, RiskLevel] = field(
        default_factory=lambda: {
            "mel": RiskLevel.HIGH,       # melanoma
            "bcc": RiskLevel.HIGH,       # basal cell carcinoma
            "akiec": RiskLevel.HIGH,     # actinic keratoses / intraepithelial carcinoma
            "scc": RiskLevel.HIGH,       # squamous cell carcinoma, if present

            "bkl": RiskLevel.MODERATE,   # benign keratosis-like lesions

            "df": RiskLevel.LOW,         # dermatofibroma
            "nv": RiskLevel.LOW,         # melanocytic nevus
            "vasc": RiskLevel.LOW,       # vascular lesions
        }
    )

    # General confidence thresholds.
    min_acceptable_top1_confidence: float = 0.45
    min_acceptable_margin: float = 0.15

    # High-risk rules.
    high_risk_urgent_threshold: float = 0.70
    high_risk_review_threshold: float = 0.50
    high_risk_topk_review_threshold: float = 0.20
    high_risk_topk: int = 2

    # Moderate-risk rules.
    moderate_risk_review_threshold: float = 0.50

    # Image-quality rules.
    fail_on_image_quality_status_fail: bool = True
    min_image_quality_score: float = 0.50

    # Grad-CAM rules.
    gradcam_min_focus_in_lesion: float = 0.50
    gradcam_warn_focus_in_lesion: float = 0.65

    # Output settings.
    top_k_to_show: int = 3


@dataclass
class PredictionItem:
    class_name: str
    probability: float
    risk_level: RiskLevel
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "class_name": self.class_name,
            "probability": self.probability,
            "risk_level": self.risk_level.value,
        }


@dataclass
class DecisionResult:
    top_predictions: List[PredictionItem]

    model_top1: PredictionItem

    risk_level: RiskLevel
    action: DecisionAction

    risk_triggering_prediction: Optional[PredictionItem]
    risk_triggering_reason: Optional[str]

    top1_probability: float
    risk_triggering_probability: Optional[float]

    margin_top1_top2: Optional[float]

    reasons: List[str]
    warnings: List[str]

    practitioner_message: str
    explanation_message: str

    gradcam_used_for_decision: Optional[GradCAMAssessment]
    gradcam_available: List[GradCAMAssessment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_predictions": [item.to_dict() for item in self.top_predictions],

            "model_top1": self.model_top1.to_dict(),

            "risk_level": self.risk_level.value,
            "action": self.action.value,

            "risk_triggering_prediction": (
                self.risk_triggering_prediction.to_dict()
                if self.risk_triggering_prediction is not None
                else None
            ),
            "risk_triggering_reason": self.risk_triggering_reason,

            "top1_probability": self.top1_probability,
            "risk_triggering_probability": self.risk_triggering_probability,

            "margin_top1_top2": self.margin_top1_top2,

            "reasons": self.reasons,
            "warnings": self.warnings,

            "practitioner_message": self.practitioner_message,
            "explanation_message": self.explanation_message,

            "gradcam_used_for_decision": (
                self.gradcam_used_for_decision.to_dict()
                if self.gradcam_used_for_decision is not None
                else None
            ),
            "gradcam_available": [
                gradcam.to_dict()
                for gradcam in self.gradcam_available
            ],
        }


def _normalise_probabilities(
    probabilities: Mapping[str, float],
) -> Dict[str, float]:
    if not probabilities:
        raise ValueError("No probabilities supplied.")

    cleaned: Dict[str, float] = {}

    for class_name, prob in probabilities.items():
        if prob is None:
            raise ValueError(f"Probability for class '{class_name}' is None.")

        prob = float(prob)

        if prob < 0:
            raise ValueError(
                f"Probability for class '{class_name}' is negative: {prob}"
            )

        cleaned[str(class_name)] = prob

    total = sum(cleaned.values())

    if total <= 0:
        raise ValueError("Probability sum must be greater than zero.")

    return {
        class_name: prob / total
        for class_name, prob in cleaned.items()
    }


def _risk_for_class(
    class_name: str,
    config: DecisionConfig,
) -> RiskLevel:
    return config.class_risk_map.get(class_name, RiskLevel.UNKNOWN)


def _raise_action(
    current: DecisionAction,
    candidate: DecisionAction,
) -> DecisionAction:
    if ACTION_RANK[candidate] > ACTION_RANK[current]:
        return candidate

    return current


def _max_risk(
    current: RiskLevel,
    candidate: RiskLevel,
) -> RiskLevel:
    if RISK_RANK[candidate] > RISK_RANK[current]:
        return candidate

    return current


def _select_relevant_gradcam(
    gradcams: List[GradCAMAssessment],
    risk_triggering_prediction: Optional[PredictionItem],
    model_top1: PredictionItem,
) -> Optional[GradCAMAssessment]:
    """
    Pick the most relevant Grad-CAM for decision explanation.

    Priority:
      1. Grad-CAM for risk-triggering class
      2. Grad-CAM for model top-1 class
      3. First available Grad-CAM
    """

    if not gradcams:
        return None

    if risk_triggering_prediction is not None:
        for gradcam in gradcams:
            if gradcam.target_class_name == risk_triggering_prediction.class_name:
                return gradcam

    for gradcam in gradcams:
        if gradcam.target_class_name == model_top1.class_name:
            return gradcam

    return gradcams[0]


def decide_from_probabilities(
    probabilities: Mapping[str, float],
    image_quality: Optional[ImageQualityAssessment] = None,
    gradcam: Optional[GradCAMAssessment] = None,
    gradcams: Optional[List[GradCAMAssessment]] = None,
    config: Optional[DecisionConfig] = None,
) -> DecisionResult:
    """
    Main decision-layer function.

    Inputs:
      probabilities:
        Dict of class_name -> probability.
        Preferably calibrated probabilities after temperature scaling.

      image_quality:
        Optional image quality assessment.

      gradcam:
        Optional single Grad-CAM assessment. Kept for backward compatibility.

      gradcams:
        Optional list of Grad-CAM assessments. Preferred going forward.

      config:
        SME-controlled rule configuration.

    Output:
      DecisionResult suitable for API/UI.
    """

    config = config or DecisionConfig()
    image_quality = image_quality or ImageQualityAssessment()

    gradcam_list: List[GradCAMAssessment] = []

    if gradcams:
        gradcam_list.extend(gradcams)

    if gradcam is not None:
        gradcam_list.append(gradcam)

    probs = _normalise_probabilities(probabilities)

    sorted_items: List[PredictionItem] = [
        PredictionItem(
            rank=rank,
            class_name=class_name,
            probability=prob,
            risk_level=_risk_for_class(class_name, config),
        )
        for rank, (class_name, prob) in enumerate(
            sorted(
                probs.items(),
                key=lambda kv: kv[1],
                reverse=True,
            ),
            start=1,
        )
    ]

    top_predictions = sorted_items[: config.top_k_to_show]

    top1 = sorted_items[0]
    top2 = sorted_items[1] if len(sorted_items) > 1 else None

    margin = None
    if top2 is not None:
        margin = top1.probability - top2.probability

    reasons: List[str] = []
    warnings: List[str] = []

    risk_triggering_prediction: Optional[PredictionItem] = None
    risk_triggering_reason: Optional[str] = None

    # ------------------------------------------------------------
    # Rule 1: Image quality override
    # ------------------------------------------------------------

    quality_status = image_quality.status.lower().strip()

    if quality_status == "fail" and config.fail_on_image_quality_status_fail:
        reasons.append("Image quality check failed.")
        reasons.extend(image_quality.notes)

        selected_gradcam = _select_relevant_gradcam(
            gradcams=gradcam_list,
            risk_triggering_prediction=None,
            model_top1=top1,
        )

        return DecisionResult(
            top_predictions=top_predictions,
            model_top1=top1,
            risk_level=RiskLevel.UNKNOWN,
            action=DecisionAction.RETAKE_IMAGE,
            risk_triggering_prediction=None,
            risk_triggering_reason=None,
            top1_probability=top1.probability,
            risk_triggering_probability=None,
            margin_top1_top2=margin,
            reasons=reasons,
            warnings=warnings,
            practitioner_message=(
                "Image quality is insufficient. Please retake the image "
                "before using model guidance."
            ),
            explanation_message=(
                "The model output should not be relied on because the input "
                "image failed quality checks."
            ),
            gradcam_used_for_decision=selected_gradcam,
            gradcam_available=gradcam_list,
        )

    if image_quality.score is not None and image_quality.score < config.min_image_quality_score:
        reasons.append(
            f"Image quality score is low: {image_quality.score:.2f}."
        )
        reasons.extend(image_quality.notes)

        selected_gradcam = _select_relevant_gradcam(
            gradcams=gradcam_list,
            risk_triggering_prediction=None,
            model_top1=top1,
        )

        return DecisionResult(
            top_predictions=top_predictions,
            model_top1=top1,
            risk_level=RiskLevel.UNKNOWN,
            action=DecisionAction.RETAKE_IMAGE,
            risk_triggering_prediction=None,
            risk_triggering_reason=None,
            top1_probability=top1.probability,
            risk_triggering_probability=None,
            margin_top1_top2=margin,
            reasons=reasons,
            warnings=warnings,
            practitioner_message=(
                "Image quality appears low. Please retake the image "
                "before using model guidance."
            ),
            explanation_message=(
                "The decision layer stopped interpretation because the image "
                "quality score was below the configured threshold."
            ),
            gradcam_used_for_decision=selected_gradcam,
            gradcam_available=gradcam_list,
        )

    if quality_status == "warn":
        warnings.append("Image quality warning present.")
        warnings.extend(image_quality.notes)

    # ------------------------------------------------------------
    # Rule 2: Base action from top-1 risk class
    # ------------------------------------------------------------

    risk_level = top1.risk_level
    risk_triggering_prediction = top1
    risk_triggering_reason = "model_top1_prediction"

    if top1.risk_level == RiskLevel.HIGH:
        if top1.probability >= config.high_risk_urgent_threshold:
            action = DecisionAction.URGENT_SPECIALIST_REVIEW
            reasons.append(
                f"Top prediction '{top1.class_name}' is high-risk "
                f"with probability {top1.probability:.2f}."
            )
        elif top1.probability >= config.high_risk_review_threshold:
            action = DecisionAction.SPECIALIST_REVIEW
            reasons.append(
                f"Top prediction '{top1.class_name}' is high-risk "
                f"with probability {top1.probability:.2f}."
            )
        else:
            action = DecisionAction.PRACTITIONER_REVIEW
            reasons.append(
                f"Top prediction '{top1.class_name}' is high-risk, "
                "but confidence is below the configured review threshold."
            )

    elif top1.risk_level == RiskLevel.MODERATE:
        if top1.probability >= config.moderate_risk_review_threshold:
            action = DecisionAction.PRACTITIONER_REVIEW
            reasons.append(
                f"Top prediction '{top1.class_name}' is moderate-risk "
                f"with probability {top1.probability:.2f}."
            )
        else:
            action = DecisionAction.ROUTINE_FOLLOWUP
            reasons.append(
                f"Top prediction '{top1.class_name}' is moderate-risk, "
                "but confidence is not high."
            )

    elif top1.risk_level == RiskLevel.LOW:
        action = DecisionAction.LOW_RISK_MONITOR
        reasons.append(
            f"Top prediction '{top1.class_name}' is low-risk "
            f"with probability {top1.probability:.2f}."
        )

    else:
        action = DecisionAction.PRACTITIONER_REVIEW
        warnings.append(
            f"Top prediction '{top1.class_name}' is not mapped to a known risk level."
        )

    # ------------------------------------------------------------
    # Rule 3: High-risk class in top-k
    # ------------------------------------------------------------

    high_risk_candidates = sorted_items[: config.high_risk_topk]

    triggered_high_risk_candidates = [
        item
        for item in high_risk_candidates
        if (
            item.risk_level == RiskLevel.HIGH
            and item.probability >= config.high_risk_topk_review_threshold
        )
    ]

    if triggered_high_risk_candidates:
        # Pick the highest probability high-risk trigger.
        high_risk_trigger = max(
            triggered_high_risk_candidates,
            key=lambda item: item.probability,
        )

        risk_level = _max_risk(risk_level, RiskLevel.HIGH)
        risk_triggering_prediction = high_risk_trigger
        risk_triggering_reason = (
            f"high_risk_class_in_top_{config.high_risk_topk}_"
            f"above_{config.high_risk_topk_review_threshold:.2f}"
        )

        if high_risk_trigger.probability >= config.high_risk_urgent_threshold:
            action = _raise_action(
                action,
                DecisionAction.URGENT_SPECIALIST_REVIEW,
            )
        else:
            action = _raise_action(
                action,
                DecisionAction.SPECIALIST_REVIEW,
            )

        if high_risk_trigger.class_name != top1.class_name:
            reasons.append(
                f"High-risk class '{high_risk_trigger.class_name}' appears in "
                f"top-{config.high_risk_topk} with probability "
                f"{high_risk_trigger.probability:.2f}."
            )
        else:
            reasons.append(
                f"High-risk top prediction '{high_risk_trigger.class_name}' "
                f"crosses the safety threshold of "
                f"{config.high_risk_topk_review_threshold:.2f}."
            )

    # ------------------------------------------------------------
    # Rule 4: Low confidence or low margin
    # ------------------------------------------------------------

    if top1.probability < config.min_acceptable_top1_confidence:
        warnings.append(
            f"Top prediction confidence is low: {top1.probability:.2f}."
        )
        action = _raise_action(action, DecisionAction.PRACTITIONER_REVIEW)

    if margin is not None and margin < config.min_acceptable_margin:
        warnings.append(
            f"Top-1 and top-2 margin is low: {margin:.2f}."
        )
        action = _raise_action(action, DecisionAction.PRACTITIONER_REVIEW)

    # ------------------------------------------------------------
    # Rule 5: Grad-CAM trust check
    # ------------------------------------------------------------

    selected_gradcam = _select_relevant_gradcam(
        gradcams=gradcam_list,
        risk_triggering_prediction=risk_triggering_prediction,
        model_top1=top1,
    )

    if gradcam_list:
        available_targets = [
            gradcam_item.target_class_name
            for gradcam_item in gradcam_list
            if gradcam_item.target_class_name is not None
        ]

        if available_targets:
            warnings.append(
                "Grad-CAM available for classes: "
                + ", ".join(available_targets)
                + "."
            )

    if selected_gradcam is not None:
        if selected_gradcam.target_class_name:
            warnings.append(
                f"Grad-CAM selected for decision explanation: "
                f"{selected_gradcam.target_class_name}."
            )

        if selected_gradcam.overlay_path:
            warnings.append(
                f"Grad-CAM overlay available at: {selected_gradcam.overlay_path}"
            )

        if selected_gradcam.heatmap_path:
            warnings.append(
                f"Grad-CAM heatmap available at: {selected_gradcam.heatmap_path}"
            )

        if selected_gradcam.artifact_focus:
            warnings.append(
                "Grad-CAM appears to focus on artifact/background rather than the lesion."
            )
            warnings.extend(selected_gradcam.notes)

            # Do not reduce risk. Only increase caution.
            action = _raise_action(action, DecisionAction.PRACTITIONER_REVIEW)

        if selected_gradcam.focus_in_lesion_score is not None:
            focus = selected_gradcam.focus_in_lesion_score

            if focus < config.gradcam_min_focus_in_lesion:
                warnings.append(
                    f"Grad-CAM lesion focus is low: {focus:.2f}."
                )
                action = _raise_action(action, DecisionAction.PRACTITIONER_REVIEW)

            elif focus < config.gradcam_warn_focus_in_lesion:
                warnings.append(
                    f"Grad-CAM lesion focus is only moderate: {focus:.2f}."
                )
        else:
            warnings.append(
                "No lesion mask supplied; Grad-CAM focus_in_lesion_score was not computed."
            )

    practitioner_message = _build_practitioner_message(
        risk_level=risk_level,
        action=action,
    )

    explanation_message = _build_explanation_message(
        top1=top1,
        top2=top2,
        margin=margin,
        risk_triggering_prediction=risk_triggering_prediction,
        risk_triggering_reason=risk_triggering_reason,
        gradcam=selected_gradcam,
    )

    return DecisionResult(
        top_predictions=top_predictions,
        model_top1=top1,
        risk_level=risk_level,
        action=action,
        risk_triggering_prediction=risk_triggering_prediction,
        risk_triggering_reason=risk_triggering_reason,
        top1_probability=top1.probability,
        risk_triggering_probability=(
            risk_triggering_prediction.probability
            if risk_triggering_prediction is not None
            else None
        ),
        margin_top1_top2=margin,
        reasons=reasons,
        warnings=warnings,
        practitioner_message=practitioner_message,
        explanation_message=explanation_message,
        gradcam_used_for_decision=selected_gradcam,
        gradcam_available=gradcam_list,
    )


def _build_practitioner_message(
    risk_level: RiskLevel,
    action: DecisionAction,
) -> str:
    if action == DecisionAction.RETAKE_IMAGE:
        return "Retake image before using model guidance."

    if action == DecisionAction.URGENT_SPECIALIST_REVIEW:
        return "High-risk finding. Urgent specialist review is recommended."

    if action == DecisionAction.SPECIALIST_REVIEW:
        return "Specialist review is recommended."

    if action == DecisionAction.PRACTITIONER_REVIEW:
        return "Model result needs practitioner review before action."

    if action == DecisionAction.ROUTINE_FOLLOWUP:
        return "Routine follow-up or practitioner review may be considered."

    if action == DecisionAction.LOW_RISK_MONITOR:
        return "Low-risk model output. Continue normal clinical judgement and monitoring."

    return "Use practitioner judgement."


def _build_explanation_message(
    top1: PredictionItem,
    top2: Optional[PredictionItem],
    margin: Optional[float],
    risk_triggering_prediction: Optional[PredictionItem],
    risk_triggering_reason: Optional[str],
    gradcam: Optional[GradCAMAssessment],
) -> str:
    parts: List[str] = []

    parts.append(
        f"The model's top prediction is '{top1.class_name}' "
        f"with probability {top1.probability:.2f}."
    )

    if top2 is not None and margin is not None:
        parts.append(
            f"The second prediction is '{top2.class_name}' "
            f"with probability {top2.probability:.2f}; "
            f"margin is {margin:.2f}."
        )

    if risk_triggering_prediction is not None:
        if risk_triggering_prediction.class_name == top1.class_name:
            parts.append(
                f"The decision is triggered by the top-1 class "
                f"'{risk_triggering_prediction.class_name}' "
                f"with probability {risk_triggering_prediction.probability:.2f}."
            )
        else:
            parts.append(
                f"The decision is triggered by '{risk_triggering_prediction.class_name}' "
                f"with probability {risk_triggering_prediction.probability:.2f}, "
                f"even though it is not the top-1 class."
            )

        if risk_triggering_reason:
            parts.append(
                f"Risk trigger reason: {risk_triggering_reason}."
            )

    if gradcam is None:
        parts.append(
            "No Grad-CAM assessment was supplied. The practitioner should review the image directly."
        )
    else:
        if gradcam.target_class_name:
            parts.append(
                f"Grad-CAM was selected for class '{gradcam.target_class_name}'."
            )

        if gradcam.focus_in_lesion_score is not None:
            parts.append(
                f"Grad-CAM lesion-focus score is {gradcam.focus_in_lesion_score:.2f}."
            )
        else:
            parts.append(
                "Grad-CAM overlay was generated, but no lesion mask was supplied; "
                "therefore a lesion-focus score was not computed."
            )

        if gradcam.artifact_focus:
            parts.append(
                "Grad-CAM may be focusing on artifacts/background, so the prediction should be interpreted cautiously."
            )
        else:
            parts.append(
                "Grad-CAM should be checked to confirm that the highlighted region corresponds to the lesion of concern."
            )

    return " ".join(parts)


def decide_from_sequence(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    image_quality: Optional[ImageQualityAssessment] = None,
    gradcam: Optional[GradCAMAssessment] = None,
    gradcams: Optional[List[GradCAMAssessment]] = None,
    config: Optional[DecisionConfig] = None,
) -> DecisionResult:
    """
    Helper when model output is a list/array rather than a dict.
    """

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Length mismatch: probabilities and class_names must have same length."
        )

    prob_dict = {
        class_name: float(prob)
        for class_name, prob in zip(class_names, probabilities)
    }

    return decide_from_probabilities(
        probabilities=prob_dict,
        image_quality=image_quality,
        gradcam=gradcam,
        gradcams=gradcams,
        config=config,
    )


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read DermAssist prediction JSON from stdin and emit decision-layer JSON."
        )
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    parser.add_argument(
        "--image-quality-status",
        default="pass",
        choices=["pass", "warn", "fail"],
        help="Optional image quality status.",
    )

    parser.add_argument(
        "--image-quality-score",
        type=float,
        default=None,
        help="Optional image quality score between 0 and 1.",
    )

    parser.add_argument(
        "--image-quality-note",
        action="append",
        default=[],
        help="Optional image quality note. Can be repeated.",
    )

    parser.add_argument(
        "--gradcam-focus-in-lesion-score",
        type=float,
        default=None,
        help="Optional Grad-CAM lesion focus score between 0 and 1.",
    )

    parser.add_argument(
        "--gradcam-artifact-focus",
        action="store_true",
        help="Flag if Grad-CAM appears to focus on artifact/background.",
    )

    parser.add_argument(
        "--gradcam-note",
        action="append",
        default=[],
        help="Optional Grad-CAM note. Can be repeated.",
    )

    return parser.parse_args()


def _read_prediction_json_from_stdin() -> Dict[str, Any]:
    raw = sys.stdin.read().strip()

    if not raw:
        raise ValueError(
            "No JSON received on stdin. Pipe output from predict_single_image.py "
            "into this script."
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON received on stdin: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must be an object.")

    return payload


def _extract_probabilities(payload: Dict[str, Any]) -> Dict[str, float]:
    """
    Supports JSON emitted by predict_single_image.py:

    {
      "schema_version": "dermassist.prediction.v1",
      "probabilities": {
        "mel": 0.62,
        "nv": 0.24
      }
    }

    Also supports raw probability dict:

    {
      "mel": 0.62,
      "nv": 0.24
    }
    """

    if "probabilities" in payload:
        probabilities = payload["probabilities"]

        if not isinstance(probabilities, dict):
            raise ValueError(
                "'probabilities' must be a dict of class_name -> probability."
            )

        return {
            str(class_name): float(probability)
            for class_name, probability in probabilities.items()
        }

    try:
        return {
            str(class_name): float(probability)
            for class_name, probability in payload.items()
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Input JSON must either contain a 'probabilities' object or be a raw "
            "dict of class_name -> probability."
        ) from exc


def _gradcam_from_item(item: Dict[str, Any]) -> GradCAMAssessment:
    focus_score = item.get("focus_in_lesion_score")

    if focus_score is not None:
        focus_score = float(focus_score)

    target_probability = item.get("target_probability")

    if target_probability is not None:
        target_probability = float(target_probability)

    notes = item.get("notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)]

    selection_reasons = item.get("selection_reasons", [])
    if not isinstance(selection_reasons, list):
        selection_reasons = [str(selection_reasons)]

    return GradCAMAssessment(
        target_class_name=item.get("target_class_name"),
        target_probability=target_probability,
        target_layer=item.get("target_layer"),
        focus_in_lesion_score=focus_score,
        artifact_focus=bool(item.get("artifact_focus", False)),
        heatmap_path=item.get("heatmap_path"),
        overlay_path=item.get("overlay_path"),
        selection_reasons=selection_reasons,
        notes=notes,
    )


def _extract_gradcams_from_payload(
    payload: Dict[str, Any],
) -> List[GradCAMAssessment]:
    """
    Supports new multi Grad-CAM shape:

    {
      "explainability": {
        "method": "grad_cam_multi",
        "items": [
          {
            "method": "grad_cam",
            "target_class_name": "bkl",
            ...
          },
          {
            "method": "grad_cam",
            "target_class_name": "mel",
            ...
          }
        ]
      }
    }

    Also supports old single Grad-CAM shape:

    {
      "explainability": {
        "method": "grad_cam",
        "target_class_name": "mel",
        ...
      }
    }
    """

    explainability = payload.get("explainability")

    if not isinstance(explainability, dict):
        return []

    method = explainability.get("method")

    if method == "grad_cam_multi":
        items = explainability.get("items", [])

        if not isinstance(items, list):
            return []

        return [
            _gradcam_from_item(item)
            for item in items
            if isinstance(item, dict)
        ]

    if method == "grad_cam":
        return [_gradcam_from_item(explainability)]

    return []


def main_cli() -> None:
    args = _parse_cli_args()

    payload = _read_prediction_json_from_stdin()
    probabilities = _extract_probabilities(payload)

    image_quality = ImageQualityAssessment(
        status=args.image_quality_status,
        score=args.image_quality_score,
        notes=args.image_quality_note,
    )

    # First try to read Grad-CAMs from piped prediction JSON.
    gradcams: List[GradCAMAssessment] = _extract_gradcams_from_payload(payload)

    # CLI Grad-CAM values override piped JSON if supplied.
    if (
        args.gradcam_focus_in_lesion_score is not None
        or args.gradcam_artifact_focus
        or args.gradcam_note
    ):
        gradcams = [
            GradCAMAssessment(
                focus_in_lesion_score=args.gradcam_focus_in_lesion_score,
                artifact_focus=args.gradcam_artifact_focus,
                notes=args.gradcam_note,
            )
        ]

    decision = decide_from_probabilities(
        probabilities=probabilities,
        image_quality=image_quality,
        gradcams=gradcams,
    )

    output = decision.to_dict()

    output["prediction_metadata"] = {
        "schema_version": payload.get("schema_version"),
        "image_path": payload.get("image_path"),
        "checkpoint_path": payload.get("checkpoint_path"),
        "temperature": payload.get("temperature"),
        "class_names": payload.get("class_names"),
        "model_top1_from_prediction_script": payload.get("model_top1"),
    }

    output["explainability"] = payload.get("explainability")

    print(json.dumps(output, indent=2 if args.pretty else None))


if __name__ == "__main__":
    try:
        main_cli()
    except Exception as exc:
        error_output = {
            "error": str(exc),
            "script": "40-decision_layer.py",
        }

        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)
