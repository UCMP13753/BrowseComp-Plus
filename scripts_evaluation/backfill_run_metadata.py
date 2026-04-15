import argparse
import json
from pathlib import Path


def load_ground_truth(dataset_path: Path) -> dict[str, dict[str, str]]:
    ground_truth: dict[str, dict[str, str]] = {}
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = str(row.get("query_id", "")).strip()
            if not qid:
                continue
            ground_truth[qid] = {
                "question": row.get("query", ""),
                "ground_truth_answer": row.get("answer", ""),
            }
    return ground_truth


def backfill_run_file(json_path: Path, ground_truth: dict[str, dict[str, str]]) -> bool:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    changed = False

    query_id = data.get("query_id")
    if query_id is not None:
        gt = ground_truth.get(str(query_id))
        if gt:
            if "question" not in data:
                data["question"] = gt["question"]
                changed = True
            if "ground_truth_answer" not in data:
                data["ground_truth_answer"] = gt["ground_truth_answer"]
                changed = True

    if "prediction" not in data:
        raw_prediction = data.get("prediction", "")
        if not raw_prediction:
            result_items = data.get("result", [])
            output_texts = [
                item.get("output", "")
                for item in result_items
                if item.get("type") == "output_text"
            ]
            if output_texts:
                data["prediction"] = output_texts[-1]
                changed = True

    if "termination" not in data and "status" in data:
        data["termination"] = data["status"]
        changed = True

    if not changed:
        return False

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill question/ground truth answer/prediction fields into run_*.json files."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing run_*.json files.",
    )
    parser.add_argument(
        "--dataset",
        default="data/browsecomp_plus_decrypted.jsonl",
        help="Path to decrypted dataset JSONL with query_id/query/answer.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ground_truth = load_ground_truth(dataset_path)

    updated = 0
    total = 0
    for json_path in sorted(run_dir.glob("run_*.json")):
        total += 1
        if backfill_run_file(json_path, ground_truth):
            updated += 1

    print(f"Scanned {total} files; updated {updated}.")


if __name__ == "__main__":
    main()
