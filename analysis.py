import argparse
import json
import os
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

from utils import DEFAULT_RESULTS_SUBDIR, build_output_path, ensure_dir, load_model_registry, require_numpy, write_csv, write_json


TOPICS = [
    "taxes",
    "immigration",
    "healthcare",
    "electric_scooters",
    "student_athletes",
    "remote_work",
    "spring_vs_fall",
    "beaches_vs_mountains",
    "coke_vs_pepsi",
]


def safe_load(path):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def safe_jsonl_rows(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_stage(stage_dir):
    topic = safe_load(os.path.join(stage_dir, "topic.npy"))
    personality = safe_load(os.path.join(stage_dir, "personality.npy"))
    judge_scores = safe_load(os.path.join(stage_dir, "judge_scores.npy"))
    unique_attributes1 = safe_load(os.path.join(stage_dir, "unique_attributes1.npy"))
    unique_attributes2 = safe_load(os.path.join(stage_dir, "unique_attributes2.npy"))
    conversation_rows = safe_jsonl_rows(os.path.join(stage_dir, "conversations.jsonl"))

    if topic is None and personality is None and judge_scores is None and not conversation_rows:
        return None

    row = {
        "stage_dir": stage_dir,
        "num_agents": int(len(topic)) if topic is not None else None,
        "topic_mean": float(np.mean(topic)) if topic is not None else None,
        "topic_std": float(np.std(topic)) if topic is not None else None,
        "persuadability_mean": float(np.mean(np.sum(personality, axis=0))) if personality is not None else None,
        "persuadability_std": float(np.std(np.sum(personality, axis=0))) if personality is not None else None,
    }

    if conversation_rows:
        row["conversation_count"] = len(conversation_rows)
        row["reasoning_leak_count"] = int(sum(r.get("num_reasoning_leaks", 0) > 0 for r in conversation_rows))
        row["truncated_count"] = int(sum(r.get("num_truncated_turns", 0) > 0 for r in conversation_rows))
        row["empty_public_count"] = int(sum(r.get("num_empty_public_turns", 0) > 0 for r in conversation_rows))
        row["repeated_turn_count"] = int(sum(r.get("repeated_public_turns", 0) > 0 for r in conversation_rows))
        row["quality_flag_count"] = int(sum(bool(r.get("conversation_quality_flag", False)) for r in conversation_rows))
    else:
        row["conversation_count"] = 0
        row["reasoning_leak_count"] = 0
        row["truncated_count"] = 0
        row["empty_public_count"] = 0
        row["repeated_turn_count"] = 0
        row["quality_flag_count"] = 0

    if judge_scores is not None:
        valid_judges = judge_scores[judge_scores >= 0]
        row["judge_mean"] = float(np.mean(valid_judges)) if valid_judges.size else None
        row["judge_std"] = float(np.std(valid_judges)) if valid_judges.size else None
        row["judge_valid_count"] = int(valid_judges.size)
    else:
        row["judge_mean"] = None
        row["judge_std"] = None
        row["judge_valid_count"] = 0

    if unique_attributes1 is not None and unique_attributes2 is not None and judge_scores is not None:
        pref_pairs = []
        for idx0, (pref1, pers1) in enumerate(unique_attributes1):
            for idx1, (pref2, pers2) in enumerate(unique_attributes2):
                scores = judge_scores[idx0, idx1]
                valid = scores[scores >= 0]
                if not valid.size:
                    continue
                pref_pairs.append(
                    {
                        "pref_gap": int(abs(int(pref1) - int(pref2))),
                        "persuadability_gap": int(abs(int(pers1) - int(pers2))),
                        "mean_agreement": float(np.mean(valid)),
                        "count": int(valid.size),
                    }
                )
        row["pair_summaries"] = pref_pairs
    else:
        row["pair_summaries"] = []

    return row


def flatten_pair_metrics(summary_rows):
    rows = []
    for row in summary_rows:
        for pair in row["pair_summaries"]:
            flat = {
                "stage_dir": row["stage_dir"],
                **pair,
            }
            rows.append(flat)
    return rows


def compute_global_metrics(pair_rows):
    if not pair_rows:
        return {}

    pref_gaps = np.array([row["pref_gap"] for row in pair_rows], dtype=float)
    mean_agreements = np.array([row["mean_agreement"] for row in pair_rows], dtype=float)
    persuadability_gaps = np.array([row["persuadability_gap"] for row in pair_rows], dtype=float)

    metrics = {
        "num_pair_bins": int(len(pair_rows)),
        "pref_gap_agreement_corr": float(np.corrcoef(pref_gaps, mean_agreements)[0, 1]) if len(pair_rows) > 1 else None,
        "persuadability_gap_agreement_corr": float(np.corrcoef(persuadability_gaps, mean_agreements)[0, 1]) if len(pair_rows) > 1 else None,
    }
    return metrics


def gather_experiment_rows(output_root, experiment_name, model_ids):
    rows = []
    results_root = build_output_path(output_root, DEFAULT_RESULTS_SUBDIR, experiment_name)
    for model_id in model_ids:
        model_dir = os.path.join(results_root, str(model_id))
        if not os.path.isdir(model_dir):
            continue
        for topic_dir in sorted(Path(model_dir).glob("*")):
            if not topic_dir.is_dir():
                continue
            for bias_dir in sorted(topic_dir.glob("*")):
                if not bias_dir.is_dir():
                    continue
                row = summarize_stage(str(bias_dir))
                if row is None:
                    continue
                row["model_id"] = str(model_id)
                row["topic_idx"] = int(topic_dir.name)
                row["bias_idx"] = int(bias_dir.name)
                row["topic_name"] = TOPICS[int(topic_dir.name)] if topic_dir.name.isdigit() and int(topic_dir.name) < len(TOPICS) else topic_dir.name
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize generated behavioral-coherence experiments.")
    parser.add_argument("--output-root", type=str, default=None, help="Artifact root.")
    parser.add_argument("--experiment-name", type=str, default="core", help="Experiment subdirectory under results/.")
    parser.add_argument("--model-ids", nargs="*", default=None, help="Model ids to summarize.")
    parser.add_argument("--analysis-name", type=str, default="core_summary", help="Output name.")
    args = parser.parse_args()
    require_numpy()

    models = load_model_registry()
    model_ids = args.model_ids or [str(model["id"]) for model in models]

    summary_rows = gather_experiment_rows(args.output_root, args.experiment_name, model_ids)
    pair_rows = flatten_pair_metrics(summary_rows)
    metrics = compute_global_metrics(pair_rows)

    tables_dir = build_output_path(args.output_root, "tables")
    figures_dir = build_output_path(args.output_root, "figures")
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    if summary_rows:
        write_csv(
            os.path.join(tables_dir, f"{args.analysis_name}_runs.csv"),
            [
                "stage_dir",
                "model_id",
                "topic_idx",
                "topic_name",
                "bias_idx",
                "num_agents",
                "topic_mean",
                "topic_std",
                "persuadability_mean",
                "persuadability_std",
                "conversation_count",
                "reasoning_leak_count",
                "truncated_count",
                "empty_public_count",
                "repeated_turn_count",
                "quality_flag_count",
                "judge_mean",
                "judge_std",
                "judge_valid_count",
            ],
            [
                {
                    key: row.get(key)
                    for key in [
                        "stage_dir",
                        "model_id",
                        "topic_idx",
                        "topic_name",
                        "bias_idx",
                        "num_agents",
                        "topic_mean",
                        "topic_std",
                        "persuadability_mean",
                        "persuadability_std",
                        "conversation_count",
                        "reasoning_leak_count",
                        "truncated_count",
                        "empty_public_count",
                        "repeated_turn_count",
                        "quality_flag_count",
                        "judge_mean",
                        "judge_std",
                        "judge_valid_count",
                    ]
                }
                for row in summary_rows
            ],
        )

    if pair_rows:
        write_csv(
            os.path.join(tables_dir, f"{args.analysis_name}_pairs.csv"),
            ["stage_dir", "pref_gap", "persuadability_gap", "mean_agreement", "count"],
            pair_rows,
        )

    write_json(
        os.path.join(figures_dir, f"{args.analysis_name}_metrics.json"),
        {
            "analysis_name": args.analysis_name,
            "experiment_name": args.experiment_name,
            "model_ids": model_ids,
            "num_runs": len(summary_rows),
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
