# src/predict_single_image.py

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


DEFAULT_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEFAULT_HIGH_RISK_CLASSES = ["mel", "bcc", "akiec", "scc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run single-image DermAssist prediction and emit calibrated "
            "probabilities as JSON."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint.",
    )

    parser.add_argument(
        "--temperature-file",
        required=True,
        help="Path to learned temperature file.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of output classes.",
    )

    parser.add_argument(
        "--class-names",
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="Class names in the same order as model output.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size expected by the model.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to include in JSON.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )

    # ------------------------------------------------------------
    # Grad-CAM options
    # ------------------------------------------------------------

    parser.add_argument(
        "--gradcam",
        action="store_true",
        help=(
            "Generate Grad-CAM heatmaps and overlays. "
            "By default, this generates Grad-CAM for top-1 and high-risk top-k classes."
        ),
    )

    parser.add_argument(
        "--gradcam-target-layer",
        default="layer4",
        help=(
            "Target layer for Grad-CAM. For ResNet18, 'layer4' is a good default."
        ),
    )

    parser.add_argument(
        "--gradcam-target-class",
        action="append",
        default=[],
        help=(
            "Additional class name for Grad-CAM. Can be repeated. "
            "Example: --gradcam-target-class mel --gradcam-target-class bcc"
        ),
    )

    parser.add_argument(
        "--gradcam-output-dir",
        default="outputs/explainability",
        help="Directory where Grad-CAM heatmap and overlay images are saved.",
    )

    parser.add_argument(
        "--lesion-mask",
        default=None,
        help=(
            "Optional lesion mask path. If supplied, a Grad-CAM "
            "focus_in_lesion_score is computed."
        ),
    )

    parser.add_argument(
        "--gradcam-high-risk-classes",
        nargs="+",
        default=DEFAULT_HIGH_RISK_CLASSES,
        help=(
            "High-risk classes for which Grad-CAM should be generated if they "
            "appear in top-k above threshold."
        ),
    )

    parser.add_argument(
        "--gradcam-high-risk-top-k",
        type=int,
        default=3,
        help=(
            "Look for high-risk classes within this top-k range for additional Grad-CAM."
        ),
    )

    parser.add_argument(
        "--gradcam-high-risk-threshold",
        type=float,
        default=0.20,
        help=(
            "Generate Grad-CAM for a high-risk class if its probability is "
            "above this threshold and it appears within gradcam-high-risk-top-k."
        ),
    )

    return parser.parse_args()


def log(message: str) -> None:
    """
    Log only to stderr so stdout remains clean JSON for piping.
    """

    print(message, file=sys.stderr)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        log("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_arg)


def build_model(num_classes: int) -> nn.Module:
    """
    Build the same architecture used during E4 training.

    This version assumes ResNet18 with a replaced final FC layer.
    If your E4 model used a different architecture, modify this function only.
    """

    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Supports common checkpoint formats:

    1. torch.save(model.state_dict(), path)

    2. torch.save({
           "model_state_dict": model.state_dict(),
           ...
       }, path)

    3. torch.save({
           "state_dict": model.state_dict(),
           ...
       }, path)

    4. torch.save({
           "model": model.state_dict(),
           ...
       }, path)
    """

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format. Expected a dict-like object.")

    for key in ["model_state_dict", "state_dict", "model"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Could not extract a valid state_dict from checkpoint.")

    cleaned_state_dict: Dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        cleaned_key = key

        # Handles checkpoints saved from DataParallel.
        if cleaned_key.startswith("module."):
            cleaned_key = cleaned_key[len("module.") :]

        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict


def load_model(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    model = build_model(num_classes=num_classes)

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
    )

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


def load_temperature(path: str) -> float:
    temperature_file = Path(path)

    if not temperature_file.exists():
        raise FileNotFoundError(f"Temperature file not found: {temperature_file}")

    temp_obj = torch.load(
        temperature_file,
        map_location="cpu",
    )

    if isinstance(temp_obj, dict):
        for key in ["temperature", "T", "temp"]:
            if key in temp_obj:
                value = temp_obj[key]
                break
        else:
            raise ValueError(
                "Temperature file dict must contain one of: "
                "'temperature', 'T', or 'temp'."
            )
    else:
        value = temp_obj

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()

    value = float(value)

    if value <= 0:
        raise ValueError(f"Temperature must be positive. Got {value}")

    return value


def preprocess_image(image_path: str, img_size: int) -> torch.Tensor:
    image_file = Path(image_path)

    if not image_file.exists():
        raise FileNotFoundError(f"Image file not found: {image_file}")

    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_file).convert("RGB")
    tensor = transform(image)

    # Shape: [1, 3, H, W]
    return tensor.unsqueeze(0)


def get_sorted_indices(probabilities: List[float]) -> List[int]:
    return sorted(
        range(len(probabilities)),
        key=lambda index: probabilities[index],
        reverse=True,
    )


def get_top_indices(
    probabilities: List[float],
    top_k: int,
) -> List[int]:
    top_k = max(1, min(top_k, len(probabilities)))
    return get_sorted_indices(probabilities)[:top_k]


def _add_gradcam_target(
    targets_by_class: Dict[str, Dict[str, Any]],
    class_name: str,
    class_index: int,
    probability: float,
    reason: str,
) -> None:
    if class_name not in targets_by_class:
        targets_by_class[class_name] = {
            "class_name": class_name,
            "class_index": class_index,
            "probability": float(probability),
            "selection_reasons": [],
        }

    if reason not in targets_by_class[class_name]["selection_reasons"]:
        targets_by_class[class_name]["selection_reasons"].append(reason)


def select_gradcam_targets(
    args: argparse.Namespace,
    probabilities: List[float],
    top_indices: List[int],
) -> List[Dict[str, Any]]:
    """
    Select Grad-CAM targets using this policy:

    1. Always include top-1 class.
    2. Include high-risk classes in top-k if probability >= configured threshold.
    3. Include any explicit classes supplied by --gradcam-target-class.

    This gives explainability for the class predicted by the model and for
    clinically important risk triggers such as mel even when they are not top-1.
    """

    class_names = args.class_names
    sorted_indices = get_sorted_indices(probabilities)

    targets_by_class: Dict[str, Dict[str, Any]] = {}

    # 1. Always include top-1 class.
    top1_index = top_indices[0]
    top1_class = class_names[top1_index]
    top1_prob = probabilities[top1_index]

    _add_gradcam_target(
        targets_by_class=targets_by_class,
        class_name=top1_class,
        class_index=top1_index,
        probability=top1_prob,
        reason="top1_prediction",
    )

    # 2. Include high-risk classes in configured top-k above threshold.
    high_risk_top_k = max(1, min(args.gradcam_high_risk_top_k, len(class_names)))
    high_risk_classes = set(args.gradcam_high_risk_classes)

    for index in sorted_indices[:high_risk_top_k]:
        class_name = class_names[index]
        probability = probabilities[index]

        if (
            class_name in high_risk_classes
            and probability >= args.gradcam_high_risk_threshold
        ):
            _add_gradcam_target(
                targets_by_class=targets_by_class,
                class_name=class_name,
                class_index=index,
                probability=probability,
                reason=(
                    f"high_risk_top_{high_risk_top_k}_"
                    f"probability_gte_{args.gradcam_high_risk_threshold:.2f}"
                ),
            )

    # 3. Include explicit target classes supplied by user.
    for explicit_class_name in args.gradcam_target_class:
        if explicit_class_name not in class_names:
            raise ValueError(
                f"Grad-CAM target class '{explicit_class_name}' is not in "
                f"class names: {class_names}"
            )

        explicit_index = class_names.index(explicit_class_name)
        explicit_probability = probabilities[explicit_index]

        _add_gradcam_target(
            targets_by_class=targets_by_class,
            class_name=explicit_class_name,
            class_index=explicit_index,
            probability=explicit_probability,
            reason="explicit_user_requested_target",
        )

    return list(targets_by_class.values())


def generate_gradcam_if_requested(
    args: argparse.Namespace,
    model: nn.Module,
    image_tensor: torch.Tensor,
    probabilities: List[float],
    top_indices: List[int],
) -> Optional[Dict[str, Any]]:
    if not args.gradcam:
        return None

    try:
        from gradcam_utils import generate_gradcam_artifacts
    except ImportError as exc:
        raise ImportError(
            "Grad-CAM requested, but gradcam_utils.py could not be imported. "
            "Make sure gradcam_utils.py is in the same directory as "
            "predict_single_image.py or available on PYTHONPATH."
        ) from exc

    targets = select_gradcam_targets(
        args=args,
        probabilities=probabilities,
        top_indices=top_indices,
    )

    items: List[Dict[str, Any]] = []

    for target in targets:
        target_index = int(target["class_index"])
        target_class_name = str(target["class_name"])

        # Important:
        # Do not wrap Grad-CAM in torch.no_grad().
        # Grad-CAM requires gradients.
        artifact = generate_gradcam_artifacts(
            model=model,
            input_tensor=image_tensor,
            original_image_path=args.image,
            target_class_index=target_index,
            target_class_name=target_class_name,
            target_layer_name=args.gradcam_target_layer,
            output_dir=args.gradcam_output_dir,
            lesion_mask_path=args.lesion_mask,
        )

        artifact["target_probability"] = float(target["probability"])
        artifact["selection_reasons"] = target["selection_reasons"]

        items.append(artifact)

    return {
        "method": "grad_cam_multi",
        "items": items,
        "primary_target_class_name": items[0]["target_class_name"] if items else None,
        "selection_policy": {
            "include_top1": True,
            "high_risk_classes": args.gradcam_high_risk_classes,
            "high_risk_top_k": args.gradcam_high_risk_top_k,
            "high_risk_threshold": args.gradcam_high_risk_threshold,
            "explicit_target_classes": args.gradcam_target_class,
            "target_layer": args.gradcam_target_layer,
        },
    }


def build_prediction_output(
    args: argparse.Namespace,
    probabilities: List[float],
    top_indices: List[int],
    temperature: float,
    explainability: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    top1_index = top_indices[0]
    top1_class = args.class_names[top1_index]
    top1_probability = float(probabilities[top1_index])

    return {
        "schema_version": "dermassist.prediction.v1",
        "image_path": str(Path(args.image)),
        "checkpoint_path": str(Path(args.checkpoint)),
        "temperature": float(temperature),
        "class_names": args.class_names,
        "probabilities": {
            args.class_names[i]: float(probabilities[i])
            for i in range(len(args.class_names))
        },
        "model_top1": {
            "class_name": top1_class,
            "probability": top1_probability,
        },
        "top_k": [
            {
                "class_name": args.class_names[index],
                "probability": float(probabilities[index]),
            }
            for index in top_indices
        ],
        "explainability": explainability,
    }


def main() -> None:
    args = parse_args()

    if len(args.class_names) != args.num_classes:
        raise ValueError(
            f"Number of class names {len(args.class_names)} does not match "
            f"--num-classes {args.num_classes}."
        )

    device = get_device(args.device)

    log(f"Using device: {device}")

    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=device,
    )

    temperature = load_temperature(args.temperature_file)

    image_tensor = preprocess_image(
        image_path=args.image,
        img_size=args.img_size,
    ).to(device)

    # Standard prediction does not need gradients.
    with torch.no_grad():
        logits = model(image_tensor)
        calibrated_logits = logits / temperature
        probs_tensor = torch.softmax(calibrated_logits, dim=1)[0]

    probabilities = probs_tensor.detach().cpu().tolist()

    top_indices = get_top_indices(
        probabilities=probabilities,
        top_k=args.top_k,
    )

    explainability = generate_gradcam_if_requested(
        args=args,
        model=model,
        image_tensor=image_tensor,
        probabilities=probabilities,
        top_indices=top_indices,
    )

    output = build_prediction_output(
        args=args,
        probabilities=probabilities,
        top_indices=top_indices,
        temperature=temperature,
        explainability=explainability,
    )

    # stdout must contain only JSON so it can be piped into decision_layer.py.
    print(json.dumps(output, indent=2 if args.pretty else None))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        error_output = {
            "error": str(exc),
            "script": "predict_single_image.py",
        }

        print(json.dumps(error_output, indent=2), file=sys.stderr)
        sys.exit(1)
