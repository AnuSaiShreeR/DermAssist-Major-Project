import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# -----------------------------
# Model loading
# -----------------------------

def build_resnet18(num_classes: int) -> nn.Module:
    """
    Must match the E4 model architecture.

    If your E4 script used ResNet18 with a replaced classifier head,
    this should match it.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def extract_state_dict(checkpoint_obj):
    """
    Handles common checkpoint formats:
    - raw state_dict
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - {"model": ...}
    """
    if isinstance(checkpoint_obj, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]

    return checkpoint_obj


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Removes common prefixes introduced by wrappers like DataParallel
    or custom checkpoint structures.
    """
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_e4_model(
    checkpoint_path: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    model = build_resnet18(num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = clean_state_dict_keys(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


# -----------------------------
# Dataset
# -----------------------------

def get_eval_transform(img_size: int):
    """
    Use the same validation/test transform used in E4.

    If your E4 training script used different resize/crop/normalization,
    change this to match exactly.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_loader(
    split_dir: Path,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, datasets.ImageFolder]:
    dataset = datasets.ImageFolder(
        root=str(split_dir),
        transform=get_eval_transform(img_size),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # Important: no sampler, no shuffle needed
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, dataset


# -----------------------------
# Logit collection
# -----------------------------

@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_logits = []
    all_labels = []

    model.eval()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return logits, labels


# -----------------------------
# Temperature scaler
# -----------------------------

class TemperatureScaler(nn.Module):
    """
    Learns one positive scalar temperature.

    We learn log_temperature so that:
        temperature = exp(log_temperature)

    This ensures T is always positive.
    """

    def __init__(self, init_temperature: float = 1.0):
        super().__init__()

        if init_temperature <= 0:
            raise ValueError("Initial temperature must be positive.")

        init_log_t = math.log(init_temperature)
        self.log_temperature = nn.Parameter(torch.tensor([init_log_t], dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def learn_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    init_temperature: float = 1.0,
    max_iter: int = 100,
) -> TemperatureScaler:
    """
    Learns T using validation logits and labels.

    Model weights are not involved here.
    Only T is optimized.
    """
    logits = logits.to(device)
    labels = labels.to(device)

    scaler = TemperatureScaler(init_temperature=init_temperature).to(device)
    criterion = nn.CrossEntropyLoss()

    before_loss = criterion(logits, labels).item()

    optimizer = optim.LBFGS(
        scaler.parameters(),
        lr=0.01,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits)
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    after_loss = criterion(scaler(logits), labels).item()

    print()
    print("Temperature learning completed")
    print(f"Validation NLL before calibration : {before_loss:.6f}")
    print(f"Validation NLL after calibration  : {after_loss:.6f}")
    print(f"Learned temperature T             : {scaler.temperature.item():.6f}")
    print()

    return scaler


# -----------------------------
# Metrics
# -----------------------------

@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: Optional[float] = None,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Computes useful calibration metrics.

    Accuracy may remain almost unchanged.
    NLL, Brier score and ECE should improve after calibration.
    """
    if temperature is not None:
        logits = logits / temperature

    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    confidences = torch.max(probs, dim=1).values

    labels = labels.long()

    accuracy = (preds == labels).float().mean().item()

    nll = nn.CrossEntropyLoss()(logits, labels).item()

    num_classes = logits.shape[1]
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    brier = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()

    ece = expected_calibration_error(
        confidences=confidences,
        predictions=preds,
        labels=labels,
        n_bins=n_bins,
    )

    return {
        "accuracy": accuracy,
        "nll": nll,
        "brier": brier,
        "ece": ece,
    }


def expected_calibration_error(
    confidences: torch.Tensor,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    ECE measures the gap between confidence and actual accuracy.

    Lower is better.
    """
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = torch.zeros(1)

    accuracies = predictions.eq(labels)

    for bin_idx in range(n_bins):
        lower = bin_boundaries[bin_idx]
        upper = bin_boundaries[bin_idx + 1]

        if bin_idx == n_bins - 1:
            in_bin = confidences.gt(lower) & confidences.le(upper)
        else:
            in_bin = confidences.gt(lower) & confidences.le(upper)

        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            confidence_in_bin = confidences[in_bin].mean()

            ece += torch.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def print_metrics(title: str, metrics: Dict[str, float]):
    print(title)
    print("-" * len(title))
    print(f"Accuracy : {metrics['accuracy']:.6f}")
    print(f"NLL      : {metrics['nll']:.6f}")
    print(f"Brier    : {metrics['brier']:.6f}")
    print(f"ECE      : {metrics['ece']:.6f}")
    print()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Dataset root directory containing val/ and optionally test/ folders.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to E4 trained model checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="temperature_calibration.pt",
        help="Where to save learned temperature.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of classes. HAM10000-style setup usually has 7.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size used for validation/test.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="LBFGS optimization iterations for temperature.",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    print(f"Using device: {device}")
    print(f"Loading E4 checkpoint: {args.checkpoint}")

    model = load_e4_model(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        device=device,
    )

    print("Building validation loader...")
    val_loader, val_dataset = build_loader(
        split_dir=val_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_names = val_dataset.classes

    print(f"Classes: {class_names}")
    print(f"Validation samples: {len(val_dataset)}")

    print("Collecting validation logits...")
    val_logits, val_labels = collect_logits_and_labels(
        model=model,
        loader=val_loader,
        device=device,
    )

    val_before = compute_metrics(val_logits, val_labels, temperature=None)
    print_metrics("Validation metrics before calibration", val_before)

    scaler = learn_temperature(
        logits=val_logits,
        labels=val_labels,
        device=device,
        init_temperature=1.0,
        max_iter=args.max_iter,
    )

    learned_t = scaler.temperature.item()

    val_after = compute_metrics(val_logits, val_labels, temperature=learned_t)
    print_metrics("Validation metrics after calibration", val_after)

    test_before = None
    test_after = None

    if test_dir.exists():
        print("Building test loader...")
        test_loader, test_dataset = build_loader(
            split_dir=test_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        print(f"Test samples: {len(test_dataset)}")

        print("Collecting test logits...")
        test_logits, test_labels = collect_logits_and_labels(
            model=model,
            loader=test_loader,
            device=device,
        )

        test_before = compute_metrics(test_logits, test_labels, temperature=None)
        test_after = compute_metrics(test_logits, test_labels, temperature=learned_t)

        print_metrics("Test metrics before calibration", test_before)
        print_metrics("Test metrics after calibration", test_after)
    else:
        print(f"No test directory found at {test_dir}. Skipping test evaluation.")

    output_payload = {
        "temperature": learned_t,
        "class_names": class_names,
        "num_classes": args.num_classes,
        "img_size": args.img_size,
        "checkpoint_path": args.checkpoint,
        "learned_from_split": "val",
        "val_metrics_before": val_before,
        "val_metrics_after": val_after,
        "test_metrics_before": test_before,
        "test_metrics_after": test_after,
    }

    torch.save(output_payload, args.output)

    print(f"Saved learned temperature to: {args.output}")
    print()
    print("Use at inference like this:")
    print("    logits = model(image)")
    print(f"    probs = torch.softmax(logits / {learned_t:.6f}, dim=1)")


if __name__ == "__main__":
    main()
