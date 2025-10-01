"""
This script computes evaluation metrics for LLM detection predictions.

Example:
    compute_metrics.py --model_name text_davinci_003 --detector_model gemma3:4b
"""

import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.utils import ROOT
from src.utils.utils import (
    compute_metrics,
    make_mixed_data,
    process_reply_from_chatgpt,
    save_metrics,
)


def load_predictions(path: Path) -> list[str]:
    with path.open(mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        predictions = {row["idx"]: row["response"] for row in reader}
        # sort predictions on the index
        sorted_predictions = sorted(predictions.items(), key=lambda x: int(x[0]))
        # return a list of all responses in the order of the sorted indices
        return list(map(lambda prediction: prediction[1], sorted_predictions))


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Computing metrics for detection predictions")
    parser.add_argument(
        "--model_name",
        help="Specify target llms to be detected.",
        required=True,
        choices=["chatgpt", "flan_t5_xxl", "text_davinci_003"],
    )
    parser.add_argument(
        "--detector_model",
        type=str,
        help="The name of the model used as detector",
        required=True,
    )
    parser.add_argument(
        "--save", action="store_true", help="Save the metrics to a file"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting the output file if it already exists.",
    )

    return parser.parse_args()


def main(args: Namespace):
    model_name: str = args.model_name
    detector_model: str = args.detector_model

    _test_data, test_labels, _mixed_test_pss = make_mixed_data(
        "data/common/test/test_humans.pkl",
        f"data/{model_name}/test/test_lms.pkl",
        "data/common/test/test_problem_statements.pkl",
    )

    detector_folder_name = detector_model.replace(":", "_")
    preds_folder = Path(ROOT / "results", "outfox_detection", detector_folder_name)
    preds_path = preds_folder / f"without_considering_attack_{model_name}.csv"
    metrics_path = preds_folder / f"without_considering_attack_{model_name}_metrics.csv"

    # Load predictions from the LLM
    print(f"Load LLM predictions from: `{preds_path}`")
    text_preds = load_predictions(preds_path)
    preds = [process_reply_from_chatgpt(text_pred) for text_pred in text_preds]

    # Compute the metrics using the labels and predictions
    human_rec, machine_rec, avg_rec, acc, precision, recall, f1 = compute_metrics(
        test_labels, preds
    )

    print(
        f"HumanRec: {human_rec}, MachineRec: {machine_rec}, AvgRec: {avg_rec}, Acc:{acc}, Precision:{precision}, Recall:{recall}, F1:{f1}"
    )

    if args.save:
        if metrics_path.exists() and not args.overwrite:
            print(
                f"Error: `{metrics_path}` already exists. Use --overwrite to replace it."
            )
            return
        else:
            save_metrics(
                metrics_path,
                {
                    "human_rec": human_rec,
                    "machine_rec": machine_rec,
                    "avg_rec": avg_rec,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                },
            )
            print(f"Saved metrics in `{metrics_path}`")


if __name__ == "__main__":
    main(parse_args())
