from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _short_model_name(name: str) -> str:
    if "ResNet-18" in name:
        return "ResNet-18\npretrained"
    if "AlexNet" in name:
        return "AlexNet-style\nfrom scratch"
    return name


def generate_comparison_chart(summary_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models = summary["models"]
    names = [_short_model_name(model["model"]) for model in models]
    accuracies = [model["test_accuracy"] * 100 for model in models]
    parameters = [model["trainable_parameters"] / 1_000_000 for model in models]

    fig, (accuracy_axis, params_axis) = plt.subplots(1, 2, figsize=(10, 4), dpi=160)
    colors = ["#2563eb", "#64748b"]

    accuracy_axis.bar(names, accuracies, color=colors, width=0.62)
    accuracy_axis.set_title("Test accuracy")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.set_ylim(80, 96)
    accuracy_axis.grid(axis="y", linestyle=":", alpha=0.35)
    for index, value in enumerate(accuracies):
        accuracy_axis.text(index, value + 0.35, f"{value:.2f}%", ha="center", fontsize=9)

    params_axis.bar(names, parameters, color=colors, width=0.62)
    params_axis.set_title("Trainable parameters")
    params_axis.set_ylabel("Parameters (M)")
    params_axis.set_ylim(0, max(parameters) * 1.2)
    params_axis.grid(axis="y", linestyle=":", alpha=0.35)
    for index, value in enumerate(parameters):
        params_axis.text(index, value + 0.6, f"{value:.2f}M", ha="center", fontsize=9)

    fig.suptitle("Selected CIFAR-10 run: accuracy vs. model size", fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        0.02,
        "Single executed notebook run; ResNet-18 uses ImageNet pretrained fine-tuning.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio result assets.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/selected_run_summary.json"),
        help="Path to selected-run summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/selected_run_comparison.png"),
        help="Output path for the comparison chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_comparison_chart(args.summary, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
