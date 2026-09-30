from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import copy
import json
import random
import time

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
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data/processed/ham10000")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")

    experiment_name: str = "E3_resnet18_frozen_weighted_loss_balanced_sampler"

    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 2
    random_seed: int = 42

    use_class_weighted_loss: bool = True
    use_weighted_sampler: bool = True

    # For the current improvement ladder, keep the backbone frozen.
    # Later E4 can set this to True and use differential learning rates.
    unfreeze_layer4: bool = False
    layer4_learning_rate: float = 1e-5
    fc_learning_rate_when_finetuning: float = 1e-4


CFG = Config()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Useful for repeatable learning experiments.
    # It may reduce performance slightly on some GPUs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms():
    weights = ResNet18_Weights.DEFAULT

    # ImageNet normalization expected by the pretrained ResNet18 weights.
    mean = weights.transforms().mean
    std = weights.transforms().std

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, eval_transform


def get_dataset_class_counts(dataset: datasets.ImageFolder) -> Counter:
    targets = [label for _, label in dataset.samples]
    return Counter(targets)


def compute_class_weights(
    dataset: datasets.ImageFolder,
    class_names: list[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.

    Formula:
        weight_c = total_samples / (num_classes * class_count_c)

    This makes minority-class mistakes more expensive in CrossEntropyLoss.
    """
    counts = get_dataset_class_counts(dataset)
    total = sum(counts.values())
    num_classes = len(class_names)

    weights = []
    for class_idx in range(num_classes):
        class_count = counts[class_idx]
        weight = total / (num_classes * class_count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Build a sampler so that minority classes appear more often in training batches.

    This changes batch composition. It does not change the validation/test data.
    """
    targets = [label for _, label in dataset.samples]
    class_counts = Counter(targets)

    sample_weights = [1.0 / class_counts[label] for label in targets]

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(cfg: Config):
    train_transform, eval_transform = build_transforms()

    train_ds = datasets.ImageFolder(cfg.data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(cfg.data_dir / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(cfg.data_dir / "test", transform=eval_transform)

    if cfg.use_weighted_sampler:
        sampler = build_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds, train_ds.classes


def build_model(num_classes: int, cfg: Config) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all pretrained layers first.
    for param in model.parameters():
        param.requires_grad = False

    # Replace ImageNet 1000-class classifier with HAM10000 classifier.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # E4 hook. Not active by default for this script's E1-E3 improvement ladder.
    if cfg.unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model


def build_optimizer(model: nn.Module, cfg: Config):
    if cfg.unfreeze_layer4:
        return optim.Adam(
            [
                {"params": model.layer4.parameters(), "lr": cfg.layer4_learning_rate},
                {"params": model.fc.parameters(), "lr": cfg.fc_learning_rate_when_finetuning},
            ]
        )

    return optim.Adam(model.fc.parameters(), lr=cfg.learning_rate)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()

    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_macro_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    running_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item() * inputs.size(0)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_probs.extend(probs.detach().cpu().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    epoch_weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    epoch_balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "macro_f1": epoch_macro_f1,
        "weighted_f1": epoch_weighted_f1,
        "balanced_accuracy": epoch_balanced_acc,
        "y_true": all_labels,
        "y_pred": all_preds,
        "y_prob": all_probs,
    }


def make_confusion_matrix_frames(y_true, y_pred, class_names: list[str]):
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


def get_key_class_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    return {
        class_name: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx, class_name in enumerate(class_names)
    }


def save_report(
    cfg: Config,
    class_names: list[str],
    train_counts: Counter,
    class_weights: torch.Tensor | None,
    history: list[dict],
    test_metrics: dict,
    cm_df: pd.DataFrame,
    cm_norm_df: pd.DataFrame,
    elapsed_minutes: float,
) -> None:
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    report_path = cfg.report_dir / f"{cfg.experiment_name}.md"
    cm_csv_path = cfg.report_dir / f"{cfg.experiment_name}_confusion_matrix.csv"
    cm_norm_csv_path = cfg.report_dir / f"{cfg.experiment_name}_confusion_matrix_normalized.csv"
    history_json_path = cfg.report_dir / f"{cfg.experiment_name}_history.json"

    cm_df.to_csv(cm_csv_path)
    cm_norm_df.to_csv(cm_norm_csv_path)

    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    class_metric_dict = get_key_class_metrics(
        test_metrics["y_true"], test_metrics["y_pred"], class_names
    )

    class_weights_dict = None
    if class_weights is not None:
        class_weights_dict = dict(zip(class_names, class_weights.detach().cpu().tolist()))

    report = []
    report.append(f"# {cfg.experiment_name}\n")
    report.append("## Configuration\n")
    report.append(f"- Model: ImageNet-pretrained ResNet18\n")
    report.append(f"- Backbone frozen: {not cfg.unfreeze_layer4}\n")
    report.append(f"- Unfreeze layer4: {cfg.unfreeze_layer4}\n")
    report.append(f"- Class-weighted loss: {cfg.use_class_weighted_loss}\n")
    report.append(f"- Weighted sampler: {cfg.use_weighted_sampler}\n")
    report.append(f"- Epochs: {cfg.num_epochs}\n")
    report.append(f"- Batch size: {cfg.batch_size}\n")
    report.append(f"- Learning rate: {cfg.learning_rate}\n")
    report.append(f"- Elapsed minutes: {elapsed_minutes:.2f}\n")

    report.append("\n## Training class distribution\n\n")
    for idx, class_name in enumerate(class_names):
        report.append(f"- {class_name}: {train_counts[idx]}\n")

    if class_weights_dict is not None:
        report.append("\n## Class weights\n\n")
        for class_name, weight in class_weights_dict.items():
            report.append(f"- {class_name}: {weight:.4f}\n")

    report.append("\n## Test metrics\n\n")
    report.append(f"- Accuracy: {test_metrics['accuracy']:.4f}\n")
    report.append(f"- Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}\n")
    report.append(f"- Macro F1: {test_metrics['macro_f1']:.4f}\n")
    report.append(f"- Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
    report.append(f"- Test loss: {test_metrics['loss']:.4f}\n")

    report.append("\n## Per-class metrics\n\n")
    report.append("| Class | Precision | Recall | F1 | Support |\n")
    report.append("|---|---:|---:|---:|---:|\n")
    for class_name, metrics in class_metric_dict.items():
        report.append(
            f"| {class_name} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['support']} |\n"
        )

    report.append("\n## Confusion matrix\n\n")
    report.append(cm_df.to_markdown())
    report.append("\n\n## Normalized confusion matrix, row-wise\n\n")
    report.append(cm_norm_df.round(3).to_markdown())
    report.append("\n\n## Files written\n\n")
    report.append(f"- Raw confusion matrix: `{cm_csv_path}`\n")
    report.append(f"- Normalized confusion matrix: `{cm_norm_csv_path}`\n")
    report.append(f"- Epoch history: `{history_json_path}`\n")

    report_path.write_text("".join(report), encoding="utf-8")
    print(f"\nSaved report: {report_path}")


def main() -> None:
    set_reproducibility(CFG.random_seed)

    CFG.model_dir.mkdir(parents=True, exist_ok=True)
    CFG.report_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    (
        train_loader,
        val_loader,
        test_loader,
        train_ds,
        _val_ds,
        _test_ds,
        class_names,
    ) = build_dataloaders(CFG)

    print("Classes:", class_names)

    train_counts = get_dataset_class_counts(train_ds)
    print("Training class counts:")
    for idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {train_counts[idx]}")

    model = build_model(num_classes=len(class_names), cfg=CFG).to(device)

    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_param_count:,}")
    print(f"Total parameters:     {total_param_count:,}")

    class_weights = None
    if CFG.use_class_weighted_loss:
        class_weights = compute_class_weights(train_ds, class_names, device)
        print("Class weights:")
        for class_name, weight in zip(class_names, class_weights.detach().cpu().tolist()):
            print(f"  {class_name}: {weight:.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, CFG)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_macro_f1 = -1.0
    best_epoch = 0
    history: list[dict] = []

    checkpoint_path = CFG.model_dir / f"{CFG.experiment_name}_best.pt"

    start_time = time.time()

    for epoch in range(1, CFG.num_epochs + 1):
        print(f"\nEpoch {epoch}/{CFG.num_epochs}")

        train_loss, train_acc, train_macro_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(epoch_record)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Train macro F1: {train_macro_f1:.4f}"
        )
        print(
            f"Val loss:   {val_metrics['loss']:.4f} | "
            f"Val acc:   {val_metrics['accuracy']:.4f} | "
            f"Val macro F1: {val_metrics['macro_f1']:.4f} | "
            f"Val balanced acc: {val_metrics['balanced_accuracy']:.4f}"
        )

        # Save best model by macro F1 rather than accuracy, because HAM10000 is imbalanced.
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_model_weights,
                    "class_names": class_names,
                    "model_name": "resnet18",
                    "num_classes": len(class_names),
                    "experiment_name": CFG.experiment_name,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "use_class_weighted_loss": CFG.use_class_weighted_loss,
                    "use_weighted_sampler": CFG.use_weighted_sampler,
                    "unfreeze_layer4": CFG.unfreeze_layer4,
                },
                checkpoint_path,
            )
            print(f"Saved new best checkpoint: {checkpoint_path}")

    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"\nTraining complete in {elapsed_minutes:.1f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro F1: {best_val_macro_f1:.4f}")

    model.load_state_dict(best_model_weights)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    print("\nTest metrics:")
    print(f"Test loss:          {test_metrics['loss']:.4f}")
    print(f"Test accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1:           {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1:        {test_metrics['weighted_f1']:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            test_metrics["y_true"],
            test_metrics["y_pred"],
            target_names=class_names,
            zero_division=0,
        )
    )

    cm_df, cm_norm_df = make_confusion_matrix_frames(
        test_metrics["y_true"], test_metrics["y_pred"], class_names
    )

    print("\nConfusion matrix:")
    print(cm_df)

    print("\nNormalized confusion matrix, row-wise:")
    print(cm_norm_df.round(3))

    save_report(
        cfg=CFG,
        class_names=class_names,
        train_counts=train_counts,
        class_weights=class_weights,
        history=history,
        test_metrics=test_metrics,
        cm_df=cm_df,
        cm_norm_df=cm_norm_df,
        elapsed_minutes=elapsed_minutes,
    )


if __name__ == "__main__":
    main()
