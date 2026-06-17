# Notebooks

This folder contains the executed CIFAR-10 CNN comparison notebook.

Open `cnn_cifar10_classification.ipynb` to review the training workflow, model comparison, plots, and evaluation diagnostics.

Use Python 3.10 or 3.11 and install the project dependencies from the repository root:

```bash
pip install -e ".[dev,notebook]"
```

The ResNet-18 run fine-tunes ImageNet pretrained weights. The script workflow in `src/cnn_cifar10_classification` uses a deterministic split seed and keeps validation/test transforms deterministic.
