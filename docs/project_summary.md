# Project Summary

This repository shows a PyTorch workflow for CIFAR-10 image classification, with an ImageNet-pretrained ResNet-18 fine-tuned against an AlexNet-style CNN trained from scratch.

## What To Highlight

- Reproducible project structure: package metadata, `src/`, `scripts/`, tests, and automated checks.
- Clear model comparison language: the ResNet-18 result is ImageNet-pretrained fine-tuning.
- Evaluation hygiene: training augmentation is not used for validation/test transforms.
- Reviewable outputs: selected-run metrics are available as JSON/CSV and as a generated chart.
- Practical automation: a quick FakeData smoke run verifies the training path before full CIFAR-10 training.

## Scope Notes

- The 93.31% result is one selected train/validation/test split.
- ResNet-18 uses ImageNet-pretrained weights, so the comparison includes a transfer-learning advantage.
- The project is an academic implementation focused on workflow, evaluation, and reporting.

## Discussion Points

- Why validation transforms should stay deterministic.
- Why a quick smoke run should stay separate from full GPU training.
- How scripts, tests, and machine-readable results make notebook work easier to inspect.
- How parameter count relates to accuracy, training cost, and transfer learning.
