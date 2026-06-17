import json
from pathlib import Path


def test_notebook_documents_pretrained_resnet_and_clean_validation_split():
    notebook = json.loads(Path("notebooks/cnn_cifar10_classification.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "models.ResNet18_Weights.DEFAULT" in source
    assert "valset_full" in source
    assert "transform=transform_test" in source
    assert "valset = Subset(valset_full, val_indices)" in source
    assert "seed_everything(SEED)" in source


def test_portfolio_display_assets_are_linked_and_present():
    readme = Path("README.md").read_text(encoding="utf-8")

    expected_paths = [
        "assets/selected_run_comparison.png",
        "docs/portfolio_summary.md",
        "results/selected_run_summary.json",
        "results/model_comparison.csv",
    ]
    for path in expected_paths:
        assert path in readme
        assert Path(path).exists()
