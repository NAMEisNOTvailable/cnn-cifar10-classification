from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .models import build_model, count_trainable_parameters

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class ExperimentConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("results")
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    val_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 2
    device: str | None = None
    quick: bool = False
    quick_train_size: int = 160
    quick_test_size: int = 64
    pretrained_resnet: bool = True
    models_to_run: tuple[str, ...] = ("alexnet", "resnet18")
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    save_checkpoints: bool = True


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_size: int
    val_size: int
    test_size: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform


def deterministic_train_val_indices(
    dataset_size: int,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    if dataset_size <= 1:
        raise ValueError("dataset_size must be greater than 1")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    val_size = max(1, int(dataset_size * val_fraction))
    train_size = dataset_size - val_size
    if train_size <= 0:
        raise ValueError("validation split leaves no training samples")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    return indices[:train_size], indices[train_size:]


def create_dataloaders(config: ExperimentConfig) -> DataLoaders:
    train_transform, eval_transform = build_transforms()

    if config.quick:
        train_full = datasets.FakeData(
            size=config.quick_train_size,
            image_size=(3, 32, 32),
            num_classes=len(CIFAR10_CLASSES),
            transform=train_transform,
        )
        val_full = datasets.FakeData(
            size=config.quick_train_size,
            image_size=(3, 32, 32),
            num_classes=len(CIFAR10_CLASSES),
            transform=eval_transform,
        )
        test_dataset = datasets.FakeData(
            size=config.quick_test_size,
            image_size=(3, 32, 32),
            num_classes=len(CIFAR10_CLASSES),
            transform=eval_transform,
        )
    else:
        train_full = datasets.CIFAR10(
            root=str(config.data_dir),
            train=True,
            download=True,
            transform=train_transform,
        )
        val_full = datasets.CIFAR10(
            root=str(config.data_dir),
            train=True,
            download=True,
            transform=eval_transform,
        )
        test_dataset = datasets.CIFAR10(
            root=str(config.data_dir),
            train=False,
            download=True,
            transform=eval_transform,
        )

    train_indices, val_indices = deterministic_train_val_indices(
        len(train_full),
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    loader_generator = torch.Generator().manual_seed(config.seed)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
    }

    return DataLoaders(
        train=DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        val=DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        test=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        test_size=len(test_dataset),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_grad_norm = 0.0
    num_batches = 0

    for batch_index, (inputs, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        predicted = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        squared_norm = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.data.norm(2).item() ** 2
        total_grad_norm += squared_norm**0.5
        num_batches += 1

    if total == 0 or num_batches == 0:
        raise RuntimeError("No training batches were processed")
    return running_loss / total, correct / total, total_grad_norm / num_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_index, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    if total == 0:
        raise RuntimeError("No evaluation batches were processed")
    return running_loss / total, correct / total


def _serializable_config(config: ExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    values["data_dir"] = str(values["data_dir"])
    values["output_dir"] = str(values["output_dir"])
    values["models_to_run"] = list(values["models_to_run"])
    return values


def _write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    rows = summary["results"]
    if rows:
        csv_path = output_dir / "model_comparison.csv"
        fieldnames = [
            "model",
            "pretrained",
            "best_val_accuracy",
            "test_accuracy",
            "test_loss",
            "trainable_parameters",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataloaders = create_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    results: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, float]]] = {}

    max_train_batches = config.max_train_batches if config.max_train_batches is not None else (1 if config.quick else None)
    max_eval_batches = config.max_eval_batches if config.max_eval_batches is not None else (1 if config.quick else None)

    for model_name in config.models_to_run:
        use_pretrained = model_name.lower().replace("-", "") in {"resnet18", "resnet"} and config.pretrained_resnet and not config.quick
        model = build_model(model_name, pretrained_resnet=use_pretrained).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        best_val_accuracy = -1.0
        best_state = copy.deepcopy(model.state_dict())
        history: list[dict[str, float]] = []

        for epoch in range(config.epochs):
            start_time = time.time()
            train_loss, train_accuracy, grad_norm = train_one_epoch(
                model,
                dataloaders.train,
                optimizer,
                criterion,
                device,
                max_batches=max_train_batches,
            )
            val_loss, val_accuracy = evaluate(
                model,
                dataloaders.val,
                criterion,
                device,
                max_batches=max_eval_batches,
            )
            scheduler.step(val_loss)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state = copy.deepcopy(model.state_dict())
                if config.save_checkpoints:
                    torch.save(best_state, output_dir / f"best_{model_name}.pth")

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "gradient_norm": grad_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.time() - start_time,
                }
            )

        model.load_state_dict(best_state)
        test_loss, test_accuracy = evaluate(
            model,
            dataloaders.test,
            criterion,
            device,
            max_batches=max_eval_batches,
        )
        results.append(
            {
                "model": model_name,
                "pretrained": use_pretrained,
                "best_val_accuracy": best_val_accuracy,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "trainable_parameters": count_trainable_parameters(model),
            }
        )
        histories[model_name] = history

    summary = {
        "config": _serializable_config(config),
        "device": str(device),
        "dataset": "FakeData smoke run" if config.quick else "CIFAR-10",
        "split": {
            "train_size": dataloaders.train_size,
            "val_size": dataloaders.val_size,
            "test_size": dataloaders.test_size,
            "val_fraction": config.val_fraction,
            "seed": config.seed,
            "validation_transform": "deterministic CIFAR-10 eval transform",
        },
        "results": results,
        "history": histories,
    }
    _write_outputs(summary, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIFAR-10 CNN comparison.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true", help="Use FakeData and batch limits for CI smoke checks.")
    parser.add_argument("--quick-train-size", type=int, default=160)
    parser.add_argument("--quick-test-size", type=int, default=64)
    parser.add_argument("--models", nargs="+", default=["alexnet", "resnet18"])
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-pretrained-resnet", dest="pretrained_resnet", action="store_false")
    parser.add_argument("--no-checkpoints", dest="save_checkpoints", action="store_false")
    parser.set_defaults(pretrained_resnet=True, save_checkpoints=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs = args.epochs if args.epochs is not None else (1 if args.quick else 50)
    num_workers = args.num_workers
    if args.quick and num_workers > 0:
        num_workers = 0
    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=num_workers,
        device=args.device,
        quick=args.quick,
        quick_train_size=args.quick_train_size,
        quick_test_size=args.quick_test_size,
        pretrained_resnet=args.pretrained_resnet,
        models_to_run=tuple(args.models),
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        save_checkpoints=args.save_checkpoints,
    )
    summary = run_experiment(config)
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
