from __future__ import annotations

import copy
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


# -----------------------------
# Configuration
# -----------------------------

EXPERIMENT_ID = "E4_resnet18_layer4_unfrozen_weighted_loss"

DATA_DIR = Path("data/processed/ham10000")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports") / EXPERIMENT_ID

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 15

# Layer4 is pretrained; update gently.
LAYER4_LR = 1e-5

# Classifier head is new; it can learn faster.
FC_LR = 1e-4

WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
RANDOM_SEED = 42

BEST_MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_ID}_best.pt"
LAST_MODEL_PATH = MODEL_DIR / f"{EXPERIMENT_ID}_last.pt"


# -----------------------------
# Utility functions
# -----------------------------

def set_seed(seed: int) -> None:
    """Make the run more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibility settings. These may reduce speed slightly.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dataset_exists() -> None:
    """Fail early if prepared ImageFolder dataset is missing."""
    required_paths = [
        DATA_DIR / "train",
        DATA_DIR / "val",
        DATA_DIR / "test",
    ]

    missing = [str(p) for p in required_paths if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Prepared dataset folders are missing:\n"
            + "\n".join(missing)
            + "\n\nExpected ImageFolder structure under data/processed/ham10000/.\n"
              "Run src/prepare_ham10000.py first."
        )


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder, List[str]]:
    """
    Build train/val/test DataLoaders.

    For E4:
    - Use weighted loss.
    - Do NOT use WeightedRandomSampler.
    - Use normal shuffle=True for training.
    """
    ensure_dataset_exists()

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocess.mean,
            std=preprocess.std,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preprocess.mean,
            std=preprocess.std,
        ),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_ds, train_ds.classes


def compute_class_weights(dataset: datasets.ImageFolder, class_names: List[str], device: torch.device) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Formula:
        weight_c = total_samples / (num_classes * class_count_c)

    This gives higher weight to minority classes.
    """
    targets = [label for _, label in dataset.samples]
    counts = Counter(targets)

    total = sum(counts.values())
    num_classes = len(class_names)

    weights_list = []
    for class_idx in range(num_classes):
        class_count = counts[class_idx]
        class_weight = total / (num_classes * class_count)
        weights_list.append(class_weight)

    return torch.tensor(weights_list, dtype=torch.float32).to(device)


def class_distribution(dataset: datasets.ImageFolder, class_names: List[str]) -> Dict[str, int]:
    """Return class distribution for an ImageFolder dataset."""
    targets = [label for _, label in dataset.samples]
    counts = Counter(targets)
    return {class_names[idx]: counts.get(idx, 0) for idx in range(len(class_names))}


def build_model(num_classes: int) -> nn.Module:
    """
    Build ResNet18 for E4.

    Training policy:
    - Load ImageNet-pretrained ResNet18.
    - Freeze all existing parameters.
    - Replace classifier head with Linear(512, num_classes).
    - Unfreeze layer4.
    - Keep model.fc trainable.
    """
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    # Replace ImageNet classifier head.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Unfreeze last residual stage.
    for param in model.layer4.parameters():
        param.requires_grad = True

    # model.fc is trainable by default because it was newly created.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return trainable and total parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_optimizer(model: nn.Module) -> optim.Optimizer:
    """
    Optimizer for E4.

    Different learning rates:
    - layer4: lower LR because it is pretrained.
    - fc: higher LR because it is newly initialized.
    """
    return optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": LAYER4_LR},
            {"params": model.fc.parameters(), "lr": FC_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# -----------------------------
# Training and evaluation
# -----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        _, preds = torch.max(logits, dim=1)

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels).item()
        total += batch_size

    avg_loss = running_loss / total
    avg_acc = running_corrects / total

    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    """Evaluate and return loss, metrics, y_true, y_pred."""
    model.eval()

    running_loss = 0.0
    total = 0

    y_true: List[int] = []
    y_pred: List[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        _, preds = torch.max(logits, dim=1)

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total += batch_size

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    avg_loss = running_loss / total

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    return avg_loss, metrics, y_true, y_pred


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    class_names: List[str],
    metrics: Dict[str, float],
    class_weights: torch.Tensor,
) -> None:
    """Save checkpoint."""
    torch.save(
        {
            "experiment_id": EXPERIMENT_ID,
            "epoch": epoch,
            "model_name": "resnet18",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
            "metrics": metrics,
            "class_weights": class_weights.detach().cpu().tolist(),
            "training_policy": {
                "backbone": "frozen_except_layer4",
                "layer4_trainable": True,
                "fc_trainable": True,
                "weighted_loss": True,
                "balanced_sampler": False,
                "layer4_lr": LAYER4_LR,
                "fc_lr": FC_LR,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
            },
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
        path,
    )


# -----------------------------
# Reporting
# -----------------------------

def make_confusion_dfs(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw and row-normalized confusion matrices as DataFrames."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )

    return cm_df, cm_norm_df


def df_to_markdown_safe(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to markdown if tabulate is installed;
    otherwise fall back to plain text.
    """
    try:
        return df.to_markdown()
    except ImportError:
        return "```text\n" + df.to_string() + "\n```"


def save_report(
    history: List[Dict[str, float]],
    class_names: List[str],
    train_distribution: Dict[str, int],
    class_weights: torch.Tensor,
    test_loss: float,
    test_metrics: Dict[str, float],
    y_true: List[int],
    y_pred: List[int],
    trainable_params: int,
    total_params: int,
) -> None:
    """Save markdown report, metrics CSV, classification report, and confusion matrices."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(REPORT_DIR / "training_history.csv", index=False)

    cm_df, cm_norm_df = make_confusion_dfs(y_true, y_pred, class_names)

    cm_df.to_csv(REPORT_DIR / "confusion_matrix_counts.csv")
    cm_norm_df.to_csv(REPORT_DIR / "confusion_matrix_row_normalized.csv")

    cls_report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cls_report_df = pd.DataFrame(cls_report_dict).transpose()
    cls_report_df.to_csv(REPORT_DIR / "classification_report.csv")

    report_lines: List[str] = []

    report_lines.append(f"# {EXPERIMENT_ID}")
    report_lines.append("")
    report_lines.append("## Purpose")
    report_lines.append("")
    report_lines.append(
        "Experiment E4 fine-tunes the last residual stage (`layer4`) of an "
        "ImageNet-pretrained ResNet18 while training a new HAM10000 classifier head. "
        "It uses class-weighted CrossEntropyLoss but does not use a balanced sampler."
    )
    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append("- Model: ResNet18")
    report_lines.append("- Pretrained weights: ImageNet / TorchVision `ResNet18_Weights.DEFAULT`")
    report_lines.append("- Dataset: HAM10000 prepared as ImageFolder")
    report_lines.append(f"- Classes: {', '.join(class_names)}")
    report_lines.append("- Backbone policy: frozen except `layer4`")
    report_lines.append("- Trainable layers: `layer4` and `fc`")
    report_lines.append("- Loss: class-weighted CrossEntropyLoss")
    report_lines.append("- Balanced sampler: no")
    report_lines.append(f"- Batch size: {BATCH_SIZE}")
    report_lines.append(f"- Epochs: {NUM_EPOCHS}")
    report_lines.append(f"- layer4 LR: {LAYER4_LR}")
    report_lines.append(f"- fc LR: {FC_LR}")
    report_lines.append(f"- Weight decay: {WEIGHT_DECAY}")
    report_lines.append(f"- Trainable parameters: {trainable_params:,}")
    report_lines.append(f"- Total parameters: {total_params:,}")
    report_lines.append("")

    report_lines.append("## Training class distribution")
    report_lines.append("")
    dist_df = pd.DataFrame(
        [{"class": k, "count": v} for k, v in train_distribution.items()]
    )
    report_lines.append(df_to_markdown_safe(dist_df))
    report_lines.append("")

    report_lines.append("## Class weights")
    report_lines.append("")
    weight_df = pd.DataFrame(
        {
            "class": class_names,
            "weight": class_weights.detach().cpu().tolist(),
        }
    )
    report_lines.append(df_to_markdown_safe(weight_df))
    report_lines.append("")

    report_lines.append("## Test metrics")
    report_lines.append("")
    metrics_table = pd.DataFrame(
        [
            {"metric": "test_loss", "value": test_loss},
            {"metric": "accuracy", "value": test_metrics["accuracy"]},
            {"metric": "balanced_accuracy", "value": test_metrics["balanced_accuracy"]},
            {"metric": "macro_f1", "value": test_metrics["macro_f1"]},
            {"metric": "weighted_f1", "value": test_metrics["weighted_f1"]},
        ]
    )
    report_lines.append(df_to_markdown_safe(metrics_table))
    report_lines.append("")

    report_lines.append("## Classification report")
    report_lines.append("")
    report_lines.append(df_to_markdown_safe(cls_report_df.round(4)))
    report_lines.append("")

    report_lines.append("## Confusion matrix - counts")
    report_lines.append("")
    report_lines.append(df_to_markdown_safe(cm_df))
    report_lines.append("")

    report_lines.append("## Confusion matrix - row normalized")
    report_lines.append("")
    report_lines.append(df_to_markdown_safe(cm_norm_df.round(3)))
    report_lines.append("")

    report_lines.append("## Interpretation checklist")
    report_lines.append("")
    report_lines.append("- Compare E4 against E1/E2/E3.")
    report_lines.append("- Do not judge only by accuracy.")
    report_lines.append("- Focus on macro F1, balanced accuracy, mel recall, akiec recall, bcc recall, and df/vasc behavior.")
    report_lines.append("- Check whether unfreezing layer4 improved minority-class recognition.")
    report_lines.append("- Check whether nv recall dropped too much.")
    report_lines.append("- Review confident wrong predictions separately.")
    report_lines.append("")

    (REPORT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    compact = {
        "experiment_id": EXPERIMENT_ID,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "train_distribution": train_distribution,
        "class_names": class_names,
        "class_weights": class_weights.detach().cpu().tolist(),
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(compact, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    set_seed(RANDOM_SEED)

    device = get_device()
    print(f"Experiment: {EXPERIMENT_ID}")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_ds, class_names = build_dataloaders()

    print("Classes:", class_names)

    train_distribution = class_distribution(train_ds, class_names)
    print("Training class distribution:", train_distribution)

    class_weights = compute_class_weights(train_ds, class_names, device)
    print("Class weights:", dict(zip(class_names, class_weights.detach().cpu().tolist())))

    model = build_model(num_classes=len(class_names)).to(device)

    trainable_params, total_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print("Trainable modules: layer4 + fc")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_macro_f1 = -1.0
    best_epoch = -1

    history: List[Dict[str, float]] = []

    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(
            "Val loss:   "
            f"{val_loss:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"Val balanced acc: {val_metrics['balanced_accuracy']:.4f} | "
            f"Val macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val weighted F1: {val_metrics['weighted_f1']:.4f}"
        )

        # Select best model by macro F1, not plain accuracy.
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                class_names=class_names,
                metrics=val_metrics,
                class_weights=class_weights,
            )

            print(f"Saved new best model: epoch={epoch}, val_macro_f1={best_val_macro_f1:.4f}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro F1: {best_val_macro_f1:.4f}")

    save_checkpoint(
        path=LAST_MODEL_PATH,
        model=model,
        optimizer=optimizer,
        epoch=NUM_EPOCHS,
        class_names=class_names,
        metrics=history[-1],
        class_weights=class_weights,
    )

    model.load_state_dict(best_model_weights)

    test_loss, test_metrics, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nTest metrics:")
    print(f"Test loss:          {test_loss:.4f}")
    print(f"Accuracy:           {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:           {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:        {test_metrics['weighted_f1']:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )
    )

    cm_df, cm_norm_df = make_confusion_dfs(y_true, y_pred, class_names)

    print("\nConfusion matrix - counts:")
    print(cm_df)

    print("\nConfusion matrix - row normalized:")
    print(cm_norm_df.round(3))

    save_report(
        history=history,
        class_names=class_names,
        train_distribution=train_distribution,
        class_weights=class_weights,
        test_loss=test_loss,
        test_metrics=test_metrics,
        y_true=y_true,
        y_pred=y_pred,
        trainable_params=trainable_params,
        total_params=total_params,
    )

    print(f"\nBest model saved to: {BEST_MODEL_PATH}")
    print(f"Last model saved to: {LAST_MODEL_PATH}")
    print(f"Report directory: {REPORT_DIR}")


if __name__ == "__main__":
    main()
