# Portfolio Summary

This repository is best presented as a supporting machine-learning portfolio project. It shows a clean PyTorch workflow for CIFAR-10 image classification, with an ImageNet-pretrained ResNet-18 fine-tuned against an AlexNet-style CNN trained from scratch.

## What To Highlight

- Reproducible project structure: package metadata, `src/`, `scripts/`, tests, and CI.
- Careful model comparison language: the ResNet-18 result is fine-tuning, not a from-scratch benchmark.
- Evaluation hygiene: training augmentation is not used for validation/test transforms.
- Reviewable outputs: selected-run metrics are available as JSON/CSV and as a generated chart.
- Practical CI: quick FakeData smoke run verifies the training path without requiring CIFAR-10 downloads or GPU time.

## What Not To Overclaim

- Do not present the 93.31% result as a multi-seed benchmark.
- Do not claim the comparison isolates architecture quality, because ResNet-18 uses ImageNet pretrained weights.
- Do not position this as original computer-vision research; it is a well-organised academic portfolio project.

## Interview Talking Points

- Why validation transforms should stay deterministic.
- Why a CI smoke run should be small and separate from full GPU training.
- How to make notebook work more reviewable through scripts, tests, and machine-readable results.
- Why parameter count alone is not a reliable measure of model quality.
