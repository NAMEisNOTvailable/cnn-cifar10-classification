"""Utilities for the CIFAR-10 CNN comparison project."""

from .experiment import ExperimentConfig, run_experiment
from .models import CIFAR10AlexNet, build_model, build_resnet18, count_trainable_parameters

__all__ = [
    "CIFAR10AlexNet",
    "ExperimentConfig",
    "build_model",
    "build_resnet18",
    "count_trainable_parameters",
    "run_experiment",
]
