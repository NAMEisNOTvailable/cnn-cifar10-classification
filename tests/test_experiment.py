import json

import torch
from torchvision import transforms

from cnn_cifar10_classification.experiment import (
    ExperimentConfig,
    create_dataloaders,
    deterministic_train_val_indices,
    run_experiment,
)
from cnn_cifar10_classification.models import CIFAR10AlexNet, build_resnet18


def _transform_types(transform):
    return {type(step) for step in transform.transforms}


def test_train_val_split_is_deterministic_and_disjoint():
    first_train, first_val = deterministic_train_val_indices(100, val_fraction=0.2, seed=7)
    second_train, second_val = deterministic_train_val_indices(100, val_fraction=0.2, seed=7)

    assert first_train == second_train
    assert first_val == second_val
    assert len(first_train) == 80
    assert len(first_val) == 20
    assert set(first_train).isdisjoint(first_val)


def test_validation_loader_uses_eval_transform_not_training_augmentation():
    config = ExperimentConfig(quick=True, batch_size=8, val_fraction=0.25, num_workers=0)
    dataloaders = create_dataloaders(config)

    train_transform = dataloaders.train.dataset.dataset.transform
    val_transform = dataloaders.val.dataset.dataset.transform

    assert transforms.RandomCrop in _transform_types(train_transform)
    assert transforms.RandomHorizontalFlip in _transform_types(train_transform)
    assert transforms.RandomCrop not in _transform_types(val_transform)
    assert transforms.RandomHorizontalFlip not in _transform_types(val_transform)


def test_models_return_cifar10_logits():
    inputs = torch.randn(2, 3, 32, 32)
    for model in (CIFAR10AlexNet(), build_resnet18(pretrained=False)):
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        assert outputs.shape == (2, 10)


def test_quick_experiment_writes_summary(tmp_path):
    config = ExperimentConfig(
        output_dir=tmp_path,
        quick=True,
        epochs=1,
        batch_size=8,
        num_workers=0,
        models_to_run=("resnet18",),
        max_train_batches=1,
        max_eval_batches=1,
        save_checkpoints=False,
    )

    summary = run_experiment(config)

    assert summary["dataset"] == "FakeData smoke run"
    assert summary["split"]["validation_transform"] == "deterministic CIFAR-10 eval transform"
    assert summary["results"][0]["model"] == "resnet18"
    assert 0.0 <= summary["results"][0]["test_accuracy"] <= 1.0
    saved_summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert saved_summary["results"][0]["pretrained"] is False
