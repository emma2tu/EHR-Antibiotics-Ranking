
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent

# INPUT_CSV = BASE / "followup_outputs" / "mean_embeddings_top10" / "combined_retrieval_classifier_summary_top10.csv"
# OUTPUT_DIR = INPUT_CSV.parent / "agreement_evaluation"
INPUT_CSV = BASE / "followup_outputs" / "combined_retrieval_classifier_summary_top10.csv"
OUTPUT_DIR = INPUT_CSV.parent / "agreement_evaluation"

ANTIBIOTICS = [
    "CLINDAMYCIN", "ERYTHROMYCIN", "GENTAMICIN", "LEVOFLOXACIN",
    "OXACILLIN", "TETRACYCLINE", "TRIMETHOPRIM/SULFA", "VANCOMYCIN",
]

def parse_ranking(x):
    if pd.isna(x):
        return []
    return [p.strip() for p in str(x).split(">")]

def top_true(row, top_col):
    if top_col not in row.index or pd.isna(row[top_col]):
        return np.nan
    ab = row[top_col]
    true_col = f"{ab}_true"
    if true_col not in row.index:
        return np.nan
    return int(row[true_col])

def count_effective_in_top_k(row, ranking_col, k):
    ranking = parse_ranking(row[ranking_col])
    total = 0
    for ab in ranking[:k]:
        true_col = f"{ab}_true"
        if true_col in row.index:
            total += int(row[true_col])
    return total

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = ["retrieval_top_antibiotic", "classifier_top_antibiotic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["agree_top1"] = work["retrieval_top_antibiotic"] == work["classifier_top_antibiotic"]
    work["retrieval_top1_true"] = work.apply(lambda r: top_true(r, "retrieval_top_antibiotic"), axis=1)
    work["classifier_top1_true"] = work.apply(lambda r: top_true(r, "classifier_top_antibiotic"), axis=1)
    work["retrieval_top3_true_count"] = work.apply(lambda r: count_effective_in_top_k(r, "retrieval_ranking", 3), axis=1)
    work["classifier_top3_true_count"] = work.apply(lambda r: count_effective_in_top_k(r, "classifier_ranking", 3), axis=1)

    work["retrieval_only_correct_top1"] = (
        (~work["agree_top1"]) & (work["retrieval_top1_true"] == 1) & (work["classifier_top1_true"] == 0)
    )
    work["classifier_only_correct_top1"] = (
        (~work["agree_top1"]) & (work["retrieval_top1_true"] == 0) & (work["classifier_top1_true"] == 1)
    )
    work["both_correct_top1"] = (work["retrieval_top1_true"] == 1) & (work["classifier_top1_true"] == 1)
    work["both_wrong_top1"] = (work["retrieval_top1_true"] == 0) & (work["classifier_top1_true"] == 0)

    rows = []
    for group_name, group in [
        ("all", work),
        ("agreement_cases", work[work["agree_top1"]]),
        ("disagreement_cases", work[~work["agree_top1"]]),
    ]:
        if len(group) == 0:
            continue
        rows.append({
            "group": group_name,
            "n_patients": int(len(group)),
            "fraction_of_all": float(len(group) / len(work)),
            "retrieval_top1_hit_rate": float(group["retrieval_top1_true"].mean()),
            "classifier_top1_hit_rate": float(group["classifier_top1_true"].mean()),
            "mean_retrieval_top3_true_count": float(group["retrieval_top3_true_count"].mean()),
            "mean_classifier_top3_true_count": float(group["classifier_top3_true_count"].mean()),
            "retrieval_only_correct_top1_rate": float(group["retrieval_only_correct_top1"].mean()),
            "classifier_only_correct_top1_rate": float(group["classifier_only_correct_top1"].mean()),
            "both_correct_top1_rate": float(group["both_correct_top1"].mean()),
            "both_wrong_top1_rate": float(group["both_wrong_top1"].mean()),
        })

    summary = pd.DataFrame(rows)

    case_cols = [
        "query_subject_id", "query_hadm_id", "query_stay_id", "query_row_id",
        "agree_top1", "retrieval_top_antibiotic", "classifier_top_antibiotic",
        "retrieval_top1_true", "classifier_top1_true",
        "retrieval_ranking", "classifier_ranking",
        "retrieval_top3_true_count", "classifier_top3_true_count",
        "top10_neighbor_subject_ids", "neighbor_similarities",
    ]
    case_cols = [c for c in case_cols if c in work.columns]

    summary_path = OUTPUT_DIR / "agreement_disagreement_summary.csv"
    cases_path = OUTPUT_DIR / "agreement_disagreement_patient_cases.csv"

    summary.to_csv(summary_path, index=False)
    work[case_cols].to_csv(cases_path, index=False)

    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {cases_path}")
    print("\nAgreement/disagreement summary:")
    print(summary)

if __name__ == "__main__":
    main()
