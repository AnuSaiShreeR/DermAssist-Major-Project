# gradcam_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Grad-CAM implementation for CNN classifiers.

    For ResNet18, a good default target layer is:
        model.layer4
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._forward_handle = None

    def _save_activation(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output

        def _save_gradient(grad: torch.Tensor) -> None:
            self.gradients = grad

        output.register_hook(_save_gradient)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_index: int,
    ) -> np.ndarray:
        """
        Returns normalized Grad-CAM heatmap as numpy array of shape H x W.
        Values are in [0, 1].
        """

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        self._forward_handle = self.target_layer.register_forward_hook(
            self._save_activation
        )

        try:
            logits = self.model(input_tensor)

            if logits.ndim != 2:
                raise ValueError(
                    f"Expected logits shape [batch, classes], got {tuple(logits.shape)}"
                )

            score = logits[:, target_class_index].sum()
            score.backward()

            if self.activations is None:
                raise RuntimeError("Grad-CAM activations were not captured.")

            if self.gradients is None:
                raise RuntimeError("Grad-CAM gradients were not captured.")

            activations = self.activations
            gradients = self.gradients

            weights = gradients.mean(dim=(2, 3), keepdim=True)

            cam_tensor = (weights * activations).sum(dim=1, keepdim=True)
            cam_tensor = F.relu(cam_tensor)

            cam_tensor = F.interpolate(
                cam_tensor,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            cam = cam_tensor[0, 0].detach().cpu().numpy()

            cam_min = float(cam.min())
            cam_max = float(cam.max())

            if cam_max - cam_min < 1e-8:
                return np.zeros_like(cam, dtype=np.float32)

            cam = (cam - cam_min) / (cam_max - cam_min)
            return cam.astype(np.float32)

        finally:
            if self._forward_handle is not None:
                self._forward_handle.remove()
                self._forward_handle = None


def get_module_by_name(
    model: torch.nn.Module,
    module_name: str,
) -> torch.nn.Module:
    """
    Resolve dotted module path.

    Examples:
        "layer4"
        "layer4.1.conv2"
    """

    current: Any = model

    for part in module_name.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)

    if not isinstance(current, torch.nn.Module):
        raise ValueError(f"Resolved object '{module_name}' is not a torch module.")

    return current


def _apply_jet_colormap(gray: np.ndarray) -> np.ndarray:
    """
    Lightweight JET-like colormap using only NumPy.

    Input:
        gray: H x W float array in [0, 1]

    Output:
        H x W x 3 uint8 RGB image
    """

    x = np.clip(gray, 0.0, 1.0)

    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_heatmap_and_overlay(
    original_image_path: str,
    cam_224: np.ndarray,
    output_dir: str,
    stem: str,
    alpha: float = 0.45,
) -> Dict[str, str]:
    """
    Saves:
      1. raw Grad-CAM heatmap image
      2. Grad-CAM overlay image on original input
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    original = Image.open(original_image_path).convert("RGB")
    original_np = np.asarray(original).astype(np.float32)

    cam_img = Image.fromarray((cam_224 * 255).astype(np.uint8))
    cam_resized = cam_img.resize(original.size, Image.BILINEAR)

    cam_resized_np = np.asarray(cam_resized).astype(np.float32) / 255.0

    heatmap_rgb = _apply_jet_colormap(cam_resized_np)

    overlay_np = (
        (1.0 - alpha) * original_np
        + alpha * heatmap_rgb.astype(np.float32)
    )

    overlay_np = np.clip(overlay_np, 0, 255).astype(np.uint8)

    heatmap_path = output_path / f"{stem}_gradcam_heatmap.png"
    overlay_path = output_path / f"{stem}_gradcam_overlay.png"

    Image.fromarray(heatmap_rgb).save(heatmap_path)
    Image.fromarray(overlay_np).save(overlay_path)

    return {
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }


def compute_focus_in_lesion_score(
    cam_224: np.ndarray,
    lesion_mask_path: Optional[str],
) -> Optional[float]:
    """
    Computes how much Grad-CAM energy lies inside a supplied lesion mask.

    Returns:
      None if no mask is supplied.

    Interpretation:
      1.0 = all Grad-CAM focus is inside lesion mask
      0.0 = none of the Grad-CAM focus is inside lesion mask
    """

    if lesion_mask_path is None:
        return None

    mask = Image.open(lesion_mask_path).convert("L")
    mask = mask.resize((cam_224.shape[1], cam_224.shape[0]), Image.NEAREST)

    mask_np = np.asarray(mask)
    mask_bool = mask_np > 127

    cam_sum = float(cam_224.sum())

    if cam_sum < 1e-8:
        return 0.0

    focus_inside = float(cam_224[mask_bool].sum())
    return focus_inside / cam_sum


def generate_gradcam_artifacts(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image_path: str,
    target_class_index: int,
    target_class_name: str,
    target_layer_name: str,
    output_dir: str,
    lesion_mask_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level helper used by predict_single_image.py.
    """

    target_layer = get_module_by_name(
        model=model,
        module_name=target_layer_name,
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    cam_224 = gradcam.generate(
        input_tensor=input_tensor,
        target_class_index=target_class_index,
    )

    image_stem = Path(original_image_path).stem
    stem = f"{image_stem}_{target_class_name}"

    paths = save_heatmap_and_overlay(
        original_image_path=original_image_path,
        cam_224=cam_224,
        output_dir=output_dir,
        stem=stem,
    )

    focus_score = compute_focus_in_lesion_score(
        cam_224=cam_224,
        lesion_mask_path=lesion_mask_path,
    )

    notes = [
        "Grad-CAM highlights regions that influenced the selected class prediction.",
        "Verify that the highlighted area corresponds to the lesion of concern.",
    ]

    if lesion_mask_path is None:
        notes.append(
            "No lesion mask supplied; focus_in_lesion_score was not computed."
        )

    return {
        "method": "grad_cam",
        "target_class_name": target_class_name,
        "target_class_index": int(target_class_index),
        "target_layer": target_layer_name,
        "heatmap_path": paths["heatmap_path"],
        "overlay_path": paths["overlay_path"],
        "focus_in_lesion_score": focus_score,
        "artifact_focus": False,
        "notes": notes,
    }
