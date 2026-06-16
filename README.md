# CNN Image Classification

CIFAR-10 computer vision project comparing ResNet-18 and AlexNet-style convolutional neural networks with training diagnostics, class-level evaluation, and architecture tradeoff analysis.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Task | CIFAR-10 image classification |
| Models | ResNet-18 and AlexNet-style CNN |
| Best result | ResNet-18 reached 93.31% test accuracy |
| Comparison point | Accuracy, model size, training behaviour, confusion matrix, and class-level errors |
| Main artefact | [`notebooks/cnn_cifar10_classification.ipynb`](notebooks/cnn_cifar10_classification.ipynb) |

## What This Demonstrates

- Built and trained deep-learning image classifiers on CIFAR-10.
- Compared a residual architecture against a larger AlexNet-style baseline.
- Used data augmentation, regularisation, and learning-rate scheduling.
- Evaluated model behaviour beyond headline accuracy with loss curves, confusion matrices, and class-level diagnostics.
- Explained the practical tradeoff between performance and parameter count.

## Results Summary

| Model | Test Accuracy | Approx. Parameters | Notes |
| --- | ---: | ---: | --- |
| ResNet-18 | 93.31% | 11.17M | Best accuracy and stronger generalisation |
| AlexNet-style CNN | 87.77% | 35.86M | Larger model with weaker final performance |

## Repository Structure

```text
notebooks/   Main experiment notebook
README.md    Portfolio overview and result summary
```

## Skills Shown

- CNN model training and evaluation
- PyTorch/TensorFlow-style deep-learning workflow
- Data augmentation and regularisation
- Model comparison and diagnostic reporting
- Clear technical communication for experimental results

## Status

Academic portfolio project. The repository is organised so reviewers can quickly understand the experiment before opening the notebook.
