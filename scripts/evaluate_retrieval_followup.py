"""
evaluate_retrieval_followup.py

Evaluates the follow-up similar-patient retrieval system against Pipeline 1
classifier probabilities.

Inputs:
    followup_outputs/combined_retrieval_classifier_summary_top10.csv

Outputs:
    followup_outputs/evaluation/
        retrieval_vs_classifier_by_antibiotic.csv
        ranking_hit_rates.csv
        rank_position_true_label_analysis.csv
        neighborhood_similarity_analysis.csv
        combined_score_weight_sweep.csv

Run from repo root:
    python evaluate_retrieval_followup.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
FOLLOWUP_DIR = BASE / "followup_outputs"
# INPUT_CSV = FOLLOWUP_DIR / "combined_retrieval_classifier_summary_top10.csv"
# EVAL_DIR = FOLLOWUP_DIR / "evaluation"
INPUT_CSV = FOLLOWUP_DIR / "mean_embeddings_top10/combined_retrieval_classifier_summary_top10.csv"
EVAL_DIR = FOLLOWUP_DIR / "evaluation_mean"


# ── Settings ──────────────────────────────────────────────────────────────────
ANTIBIOTICS = [
    "CLINDAMYCIN",
    "ERYTHROMYCIN",
    "GENTAMICIN",
    "LEVOFLOXACIN",
    "OXACILLIN",
    "TETRACYCLINE",
    "TRIMETHOPRIM/SULFA",
    "VANCOMYCIN",
]

# Combined score:
# combined = weight * classifier_proba + (1 - weight) * retrieval_fraction
COMBINED_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_metric(metric_func, y_true, y_score_or_pred, default=np.nan, **kwargs):
    """Compute metric safely when labels may be single-class."""
    try:
        return float(metric_func(y_true, y_score_or_pred, **kwargs))
    except Exception:
        return default


def get_antibiotic_cols(ab: str):
    """Return expected columns for one antibiotic."""
    return {
        "true": f"{ab}_true",
        "retrieval_fraction": f"{ab}_retrieval_fraction",
        "retrieval_count": f"{ab}_retrieval_count",
        "retrieval_weighted_score": f"{ab}_retrieval_weighted_score",
        "classifier_proba": f"{ab}_classifier_proba",
        "classifier_pred": f"{ab}_classifier_pred",
    }


def parse_ranking(ranking_str: str) -> list[str]:
    """Parse ranking string formatted as 'A > B > C'."""
    if pd.isna(ranking_str):
        return []
    return [x.strip() for x in str(ranking_str).split(">")]


def get_top_k_from_ranking(ranking_str: str, k: int) -> list[str]:
    return parse_ranking(ranking_str)[:k]


def row_true_labels(row: pd.Series, antibiotics: list[str]) -> dict[str, int]:
    return {
        ab: int(row[f"{ab}_true"])
        for ab in antibiotics
        if f"{ab}_true" in row.index and not pd.isna(row[f"{ab}_true"])
    }


def rank_by_score(row: pd.Series, antibiotics: list[str], score_suffix: str) -> list[str]:
    """
    Rank antibiotics descending by a column suffix.
    Example suffix: '_classifier_proba', '_retrieval_fraction', '_combined_0.50'
    """
    scored = []
    for ab in antibiotics:
        col = f"{ab}{score_suffix}"
        if col in row.index and not pd.isna(row[col]):
            scored.append((ab, float(row[col])))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return [ab for ab, _ in scored]


def top_k_hit(row: pd.Series, ranking: list[str], k: int, antibiotics: list[str]) -> bool:
    """
    True if at least one of the top-k ranked antibiotics has true label = 1.
    """
    labels = row_true_labels(row, antibiotics)
    top = ranking[:k]
    return any(labels.get(ab, 0) == 1 for ab in top)


def top1_true(row: pd.Series, ranking: list[str]) -> float:
    """Return true label of the top-ranked antibiotic."""
    if not ranking:
        return np.nan
    ab = ranking[0]
    col = f"{ab}_true"
    if col not in row.index or pd.isna(row[col]):
        return np.nan
    return float(row[col])


def average_precision_at_k(row: pd.Series, ranking: list[str], k: int, antibiotics: list[str]) -> float:
    """
    AP@k for a multi-label row:
    Rewards rankings that place true-effective antibiotics near the top.
    """
    labels = row_true_labels(row, antibiotics)
    if not labels:
        return np.nan

    total_relevant = sum(labels.values())
    if total_relevant == 0:
        return np.nan

    hits = 0
    precisions = []

    for rank_idx, ab in enumerate(ranking[:k], start=1):
        if labels.get(ab, 0) == 1:
            hits += 1
            precisions.append(hits / rank_idx)

    if not precisions:
        return 0.0

    return float(sum(precisions) / min(total_relevant, k))


def ndcg_at_k(row: pd.Series, ranking: list[str], k: int, antibiotics: list[str]) -> float:
    """
    NDCG@k for binary relevance across antibiotics.
    """
    labels = row_true_labels(row, antibiotics)
    if not labels:
        return np.nan

    def dcg(rels):
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))

    rels = [labels.get(ab, 0) for ab in ranking[:k]]
    dcg_val = dcg(rels)

    ideal_rels = sorted(labels.values(), reverse=True)[:k]
    idcg_val = dcg(ideal_rels)

    if idcg_val == 0:
        return np.nan

    return float(dcg_val / idcg_val)


def mean_neighbor_similarity(row: pd.Series) -> float:
    """
    Parse neighbor_similarities JSON string and return mean.
    """
    if "neighbor_similarities" not in row.index or pd.isna(row["neighbor_similarities"]):
        return np.nan

    try:
        sims = json.loads(row["neighbor_similarities"])
        if not sims:
            return np.nan
        return float(np.mean(sims))
    except Exception:
        return np.nan


# ── Evaluation 1: Per-antibiotic score metrics ────────────────────────────────
def evaluate_by_antibiotic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat each antibiotic score as a binary prediction for its true label.

    Compares:
      - retrieval_fraction
      - retrieval_weighted_score
      - classifier_proba
      - combined scores
    """
    rows = []

    for ab in ANTIBIOTICS:
        cols = get_antibiotic_cols(ab)

        required = [cols["true"], cols["retrieval_fraction"], cols["retrieval_weighted_score"]]
        missing_required = [c for c in required if c not in df.columns]
        if missing_required:
            print(f"Skipping {ab}; missing required columns: {missing_required}")
            continue

        y_true = df[cols["true"]].astype(int).to_numpy()

        score_columns = {
            "retrieval_fraction": cols["retrieval_fraction"],
            "retrieval_weighted_score": cols["retrieval_weighted_score"],
        }

        if cols["classifier_proba"] in df.columns:
            score_columns["classifier_proba"] = cols["classifier_proba"]

            # Create combined score columns in-memory.
            for w in COMBINED_WEIGHTS:
                method_name = f"combined_w_classifier_{w:.2f}"
                score_columns[method_name] = method_name
                df[method_name] = (
                    w * df[cols["classifier_proba"]].astype(float)
                    + (1 - w) * df[cols["retrieval_fraction"]].astype(float)
                )

        for method, score_col in score_columns.items():
            if score_col not in df.columns:
                continue

            y_score = df[score_col].astype(float).to_numpy()

            # Thresholded prediction for retrieval scores using 0.5.
            # For classifier, use saved pred if available; otherwise also threshold at 0.5.
            if method == "classifier_proba" and cols["classifier_pred"] in df.columns:
                y_pred = df[cols["classifier_pred"]].astype(int).to_numpy()
            else:
                y_pred = (y_score >= 0.5).astype(int)

            rows.append({
                "antibiotic": ab,
                "method": method,
                "auroc": safe_metric(roc_auc_score, y_true, y_score),
                "auprc": safe_metric(average_precision_score, y_true, y_score),
                "accuracy_at_0.5": safe_metric(accuracy_score, y_true, y_pred),
                "precision_at_0.5": safe_metric(precision_score, y_true, y_pred, zero_division=0),
                "recall_at_0.5": safe_metric(recall_score, y_true, y_pred, zero_division=0),
                "f1_at_0.5": safe_metric(f1_score, y_true, y_pred, zero_division=0),
                "prevalence": float(np.mean(y_true)),
                "n": int(len(y_true)),
            })

    return pd.DataFrame(rows)


# ── Evaluation 2: Ranking hit rates ───────────────────────────────────────────
def evaluate_ranking_hit_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate whether top-ranked antibiotics are truly effective.

    For each row:
      - top-1 true label
      - top-3 hit
      - AP@k
      - NDCG@k
    """
    rows = []
    methods = []

    if "retrieval_ranking" in df.columns:
        methods.append(("retrieval", "retrieval_ranking", None))

    if "classifier_ranking" in df.columns:
        methods.append(("classifier", "classifier_ranking", None))

    # Add combined ranking methods based on score columns created here.
    if all(f"{ab}_classifier_proba" in df.columns for ab in ANTIBIOTICS):
        for w in COMBINED_WEIGHTS:
            suffix = f"_combined_score_w{w:.2f}"
            for ab in ANTIBIOTICS:
                df[f"{ab}{suffix}"] = (
                    w * df[f"{ab}_classifier_proba"].astype(float)
                    + (1 - w) * df[f"{ab}_retrieval_fraction"].astype(float)
                )
            methods.append((f"combined_w_classifier_{w:.2f}", None, suffix))

    for method_name, ranking_col, score_suffix in methods:
        top1_values = []
        top3_hits = []
        top5_hits = []
        ap3_values = []
        ap5_values = []
        ndcg3_values = []
        ndcg5_values = []

        for _, row in df.iterrows():
            if ranking_col is not None:
                ranking = parse_ranking(row[ranking_col])
            else:
                ranking = rank_by_score(row, ANTIBIOTICS, score_suffix)

            top1_values.append(top1_true(row, ranking))
            top3_hits.append(top_k_hit(row, ranking, 3, ANTIBIOTICS))
            top5_hits.append(top_k_hit(row, ranking, 5, ANTIBIOTICS))
            ap3_values.append(average_precision_at_k(row, ranking, 3, ANTIBIOTICS))
            ap5_values.append(average_precision_at_k(row, ranking, 5, ANTIBIOTICS))
            ndcg3_values.append(ndcg_at_k(row, ranking, 3, ANTIBIOTICS))
            ndcg5_values.append(ndcg_at_k(row, ranking, 5, ANTIBIOTICS))

        rows.append({
            "method": method_name,
            "top1_hit_rate": float(np.nanmean(top1_values)),
            "top3_hit_rate": float(np.mean(top3_hits)),
            "top5_hit_rate": float(np.mean(top5_hits)),
            "mean_ap_at_3": float(np.nanmean(ap3_values)),
            "mean_ap_at_5": float(np.nanmean(ap5_values)),
            "mean_ndcg_at_3": float(np.nanmean(ndcg3_values)),
            "mean_ndcg_at_5": float(np.nanmean(ndcg5_values)),
            "n_patients": int(len(df)),
        })

    return pd.DataFrame(rows)


# ── Evaluation 3: True-label rate by rank position ────────────────────────────
def evaluate_rank_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each method, ask:
      Are antibiotics ranked #1 more often truly effective than #8?
    """
    rows = []
    ranking_specs = []

    if "retrieval_ranking" in df.columns:
        ranking_specs.append(("retrieval", "retrieval_ranking", None))

    if "classifier_ranking" in df.columns:
        ranking_specs.append(("classifier", "classifier_ranking", None))

    for method_name, ranking_col, _ in ranking_specs:
        rank_to_values = {rank: [] for rank in range(1, len(ANTIBIOTICS) + 1)}

        for _, row in df.iterrows():
            ranking = parse_ranking(row[ranking_col])

            for rank, ab in enumerate(ranking, start=1):
                true_col = f"{ab}_true"
                if true_col in row.index and not pd.isna(row[true_col]):
                    rank_to_values[rank].append(int(row[true_col]))

        for rank, vals in rank_to_values.items():
            rows.append({
                "method": method_name,
                "rank_position": rank,
                "mean_true_label": float(np.mean(vals)) if vals else np.nan,
                "n_values": int(len(vals)),
            })

    return pd.DataFrame(rows)


# ── Evaluation 4: Neighborhood quality ────────────────────────────────────────
def evaluate_neighborhood_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Does retrieval work better when the nearest neighbors are more similar?
    """
    work = df.copy()
    work["mean_neighbor_similarity"] = work.apply(mean_neighbor_similarity, axis=1)

    if work["mean_neighbor_similarity"].isna().all():
        return pd.DataFrame()

    # Retrieval top-1 correctness.
    work["retrieval_top1_true"] = work.apply(
        lambda row: top1_true(row, parse_ranking(row["retrieval_ranking"]))
        if "retrieval_ranking" in row.index else np.nan,
        axis=1,
    )

    # Split into quartiles of neighborhood similarity.
    work = work.dropna(subset=["mean_neighbor_similarity", "retrieval_top1_true"]).copy()

    if len(work) < 4:
        return pd.DataFrame()

    work["similarity_quartile"] = pd.qcut(
        work["mean_neighbor_similarity"],
        q=4,
        labels=["Q1_lowest_similarity", "Q2", "Q3", "Q4_highest_similarity"],
        duplicates="drop",
    )

    rows = []

    for quartile, group in work.groupby("similarity_quartile", observed=True):
        rows.append({
            "similarity_quartile": str(quartile),
            "n_patients": int(len(group)),
            "mean_neighbor_similarity": float(group["mean_neighbor_similarity"].mean()),
            "retrieval_top1_hit_rate": float(group["retrieval_top1_true"].mean()),
        })

    return pd.DataFrame(rows)


# ── Evaluation 5: Weight sweep summary ────────────────────────────────────────
def evaluate_combined_weight_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize combined score performance averaged across antibiotics
    for each classifier/retrieval mixing weight.
    """
    rows = []

    for w in COMBINED_WEIGHTS:
        per_ab = []

        for ab in ANTIBIOTICS:
            true_col = f"{ab}_true"
            clf_col = f"{ab}_classifier_proba"
            ret_col = f"{ab}_retrieval_fraction"

            if not all(c in df.columns for c in [true_col, clf_col, ret_col]):
                continue

            y_true = df[true_col].astype(int).to_numpy()
            y_score = (
                w * df[clf_col].astype(float).to_numpy()
                + (1 - w) * df[ret_col].astype(float).to_numpy()
            )

            per_ab.append({
                "antibiotic": ab,
                "auroc": safe_metric(roc_auc_score, y_true, y_score),
                "auprc": safe_metric(average_precision_score, y_true, y_score),
            })

        if not per_ab:
            continue

        per_ab_df = pd.DataFrame(per_ab)

        rows.append({
            "classifier_weight": w,
            "retrieval_weight": 1 - w,
            "mean_auroc_across_antibiotics": float(per_ab_df["auroc"].mean()),
            "mean_auprc_across_antibiotics": float(per_ab_df["auprc"].mean()),
            "median_auroc_across_antibiotics": float(per_ab_df["auroc"].median()),
            "median_auprc_across_antibiotics": float(per_ab_df["auprc"].median()),
            "n_antibiotics": int(len(per_ab_df)),
        })

    return pd.DataFrame(rows)


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Could not find follow-up summary CSV: {INPUT_CSV}\n"
            "Run retrieval_followup_from_frozen_outputs.py first."
        )

    print(f"Reading follow-up summary: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Rows: {len(df)}  Columns: {len(df.columns)}")

    by_ab = evaluate_by_antibiotic(df.copy())
    hit_rates = evaluate_ranking_hit_rates(df.copy())
    rank_positions = evaluate_rank_positions(df.copy())
    neighborhood = evaluate_neighborhood_similarity(df.copy())
    weight_sweep = evaluate_combined_weight_sweep(df.copy())

    by_ab_path = EVAL_DIR / "retrieval_vs_classifier_by_antibiotic.csv"
    hit_rates_path = EVAL_DIR / "ranking_hit_rates.csv"
    rank_positions_path = EVAL_DIR / "rank_position_true_label_analysis.csv"
    neighborhood_path = EVAL_DIR / "neighborhood_similarity_analysis.csv"
    weight_sweep_path = EVAL_DIR / "combined_score_weight_sweep.csv"

    by_ab.to_csv(by_ab_path, index=False)
    hit_rates.to_csv(hit_rates_path, index=False)
    rank_positions.to_csv(rank_positions_path, index=False)
    neighborhood.to_csv(neighborhood_path, index=False)
    weight_sweep.to_csv(weight_sweep_path, index=False)

    print("\nSaved evaluation outputs:")
    print(f"  Per-antibiotic score metrics: {by_ab_path}")
    print(f"  Ranking hit rates:           {hit_rates_path}")
    print(f"  Rank-position analysis:      {rank_positions_path}")
    print(f"  Neighborhood analysis:       {neighborhood_path}")
    print(f"  Combined weight sweep:       {weight_sweep_path}")

    print("\nQuick view: ranking hit rates")
    print(hit_rates)

    print("\nQuick view: combined weight sweep")
    print(weight_sweep)

    if not by_ab.empty:
        print("\nMean AUROC/AUPRC by method:")
        print(
            by_ab.groupby("method")[["auroc", "auprc"]]
            .mean()
            .sort_values("auroc", ascending=False)
        )


if __name__ == "__main__":
    main()
