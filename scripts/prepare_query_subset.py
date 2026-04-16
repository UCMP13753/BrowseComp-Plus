#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path


def load_queries(path: Path) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            qid = row[0].strip()
            query = row[1].strip()
            if qid and query:
                queries.append((qid, query))
    return queries


def write_queries(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerows(rows)


def filter_qrel_file(src: Path, dst: Path, selected_ids: set[str]) -> int:
    kept = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.split()
            if parts and parts[0] in selected_ids:
                fout.write(line)
                kept += 1
    return kept


def filter_ground_truth(src: Path, dst: Path, selected_ids: set[str]) -> int:
    kept = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            record = json.loads(line)
            if str(record.get("query_id", "")).strip() in selected_ids:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic subset of BrowseComp-Plus queries and aligned metadata files."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("topics-qrels/queries.tsv"),
        help="Path to the source TSV file of query_id<TAB>query.",
    )
    parser.add_argument(
        "--qrel-golds",
        type=Path,
        default=Path("topics-qrels/qrel_golds.txt"),
        help="Path to the source gold qrels file.",
    )
    parser.add_argument(
        "--qrel-evidence",
        type=Path,
        default=Path("topics-qrels/qrel_evidence.txt"),
        help="Path to the source evidence qrels file.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("data/browsecomp_plus_decrypted.jsonl"),
        help="Path to the source decrypted JSONL dataset.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of queries to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("topics-qrels/subsets/random100_seed42"),
        help="Directory to store the generated subset files.",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries)
    total_queries = len(queries)
    if total_queries == 0:
        raise ValueError(f"No queries found in {args.queries}")
    if args.size <= 0:
        raise ValueError("--size must be positive")
    if args.size > total_queries:
        raise ValueError(f"--size={args.size} exceeds available queries={total_queries}")

    rng = random.Random(args.seed)
    sampled_ids = set(rng.sample([qid for qid, _ in queries], args.size))
    sampled_rows = [(qid, query) for qid, query in queries if qid in sampled_ids]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_queries(output_dir / "queries.tsv", sampled_rows)

    with (output_dir / "query_ids.txt").open("w", encoding="utf-8") as f:
        for qid, _ in sampled_rows:
            f.write(f"{qid}\n")

    qrel_golds_count = filter_qrel_file(args.qrel_golds, output_dir / "qrel_golds.txt", sampled_ids)
    qrel_evidence_count = filter_qrel_file(
        args.qrel_evidence, output_dir / "qrel_evidence.txt", sampled_ids
    )
    ground_truth_count = filter_ground_truth(
        args.ground_truth, output_dir / "browsecomp_plus_decrypted.jsonl", sampled_ids
    )

    metadata = {
        "source_queries": str(args.queries),
        "source_qrel_golds": str(args.qrel_golds),
        "source_qrel_evidence": str(args.qrel_evidence),
        "source_ground_truth": str(args.ground_truth),
        "subset_size": args.size,
        "total_queries": total_queries,
        "seed": args.seed,
        "query_ids": [qid for qid, _ in sampled_rows],
        "counts": {
            "queries": len(sampled_rows),
            "qrel_golds_lines": qrel_golds_count,
            "qrel_evidence_lines": qrel_evidence_count,
            "ground_truth_records": ground_truth_count,
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote subset to {output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
