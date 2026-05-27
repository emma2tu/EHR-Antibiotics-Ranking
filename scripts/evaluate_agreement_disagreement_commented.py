# Agreement/disagreement evaluation for the retrieval follow-up pipeline.
# This script compares the top antibiotic recommended by the retrieval method
# against the top antibiotic recommended by the classifier probability ranking.
# It is meant to identify cases where the two methods agree, disagree, or one
# method correctly ranks an actually effective antibiotic above the other.
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Resolve paths relative to this script so the file can be run from the repo root.
BASE = Path(__file__).parent

# Alternative input/output paths from an earlier mean-embedding run.
# These are left here so the evaluation can easily be switched back if needed.
# INPUT_CSV = BASE / "followup_outputs" / "mean_embeddings_top10" / "combined_retrieval_classifier_summary_top10.csv"
# OUTPUT_DIR = INPUT_CSV.parent / "agreement_evaluation"

# Main follow-up summary produced by the retrieval + classifier comparison script.
INPUT_CSV = BASE / "followup_outputs" / "combined_retrieval_classifier_summary_top10.csv"

# Folder where this script writes the agreement/disagreement summary files.
OUTPUT_DIR = INPUT_CSV.parent / "agreement_evaluation"

# Antibiotic labels used throughout the project.
# Each antibiotic has corresponding columns such as:
#   <ANTIBIOTIC>_true
#   <ANTIBIOTIC>_retrieval_fraction
#   <ANTIBIOTIC>_classifier_proba
ANTIBIOTICS = [
    "CLINDAMYCIN", "ERYTHROMYCIN", "GENTAMICIN", "LEVOFLOXACIN",
    "OXACILLIN", "TETRACYCLINE", "TRIMETHOPRIM/SULFA", "VANCOMYCIN",
]

# Convert a stored ranking string such as:
#   "VANCOMYCIN > OXACILLIN > CLINDAMYCIN"
# into a Python list of antibiotic names.
def parse_ranking(x):
    if pd.isna(x):
        return []
    return [p.strip() for p in str(x).split(">")]

# Given a row and the column containing the top-ranked antibiotic,
# return whether that top antibiotic was truly effective for the patient.
# Returns NaN if the required column is unavailable.
def top_true(row, top_col):
    if top_col not in row.index or pd.isna(row[top_col]):
        return np.nan
    ab = row[top_col]
    true_col = f"{ab}_true"
    if true_col not in row.index:
        return np.nan
    return int(row[true_col])

# Count how many of the top-k antibiotics in a ranking have true label = 1.
# This measures whether a method places actually effective antibiotics near the top,
# even if its single top-ranked antibiotic is not correct.
def count_effective_in_top_k(row, ranking_col, k):
    ranking = parse_ranking(row[ranking_col])
    total = 0
    for ab in ranking[:k]:
        true_col = f"{ab}_true"
        if true_col in row.index:
            total += int(row[true_col])
    return total

def main():
    # Create the output directory if it does not already exist.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Stop early with a clear error if the expected retrieval/classifier summary is missing.
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    # Load the patient-level summary table that contains both retrieval and classifier rankings.
    df = pd.read_csv(INPUT_CSV)

    # These columns are required because this script compares the top antibiotic
    # selected by retrieval versus the top antibiotic selected by the classifier.
    required = ["retrieval_top_antibiotic", "classifier_top_antibiotic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Work on a copy so the original loaded dataframe is not modified directly.
    work = df.copy()

    # Agreement is defined as both methods choosing the same #1 antibiotic.
    work["agree_top1"] = work["retrieval_top_antibiotic"] == work["classifier_top_antibiotic"]

    # For each method's top antibiotic, record whether that antibiotic was truly effective.
    work["retrieval_top1_true"] = work.apply(lambda r: top_true(r, "retrieval_top_antibiotic"), axis=1)
    work["classifier_top1_true"] = work.apply(lambda r: top_true(r, "classifier_top_antibiotic"), axis=1)

    # Count how many truly effective antibiotics appear in each method's top 3 ranking.
    work["retrieval_top3_true_count"] = work.apply(lambda r: count_effective_in_top_k(r, "retrieval_ranking", 3), axis=1)
    work["classifier_top3_true_count"] = work.apply(lambda r: count_effective_in_top_k(r, "classifier_ranking", 3), axis=1)

    # Disagreement case where retrieval's top choice is correct but classifier's top choice is not.
    work["retrieval_only_correct_top1"] = (
        (~work["agree_top1"]) & (work["retrieval_top1_true"] == 1) & (work["classifier_top1_true"] == 0)
    )

    # Disagreement case where classifier's top choice is correct but retrieval's top choice is not.
    work["classifier_only_correct_top1"] = (
        (~work["agree_top1"]) & (work["retrieval_top1_true"] == 0) & (work["classifier_top1_true"] == 1)
    )

    # Cases where both top-ranked antibiotics are truly effective.
    # In agreement cases, this means both methods chose the same correct antibiotic.
    # In disagreement cases, this means both methods chose different but still effective antibiotics.
    work["both_correct_top1"] = (work["retrieval_top1_true"] == 1) & (work["classifier_top1_true"] == 1)

    # Cases where neither method's top-ranked antibiotic was truly effective.
    work["both_wrong_top1"] = (work["retrieval_top1_true"] == 0) & (work["classifier_top1_true"] == 0)

    # Build a compact summary across:
    #   1. all patients
    #   2. patients where retrieval/classifier agree on top-1
    #   3. patients where retrieval/classifier disagree on top-1
    rows = []
    for group_name, group in [
        ("all", work),
        ("agreement_cases", work[work["agree_top1"]]),
        ("disagreement_cases", work[~work["agree_top1"]]),
    ]:
        if len(group) == 0:
            continue

        # Each row summarizes how often the top choices were correct,
        # plus how many truly effective antibiotics appeared in each method's top 3.
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

    # Patient-level columns to export for manual review of agreement/disagreement cases.
    # The list is filtered below so the script still works if some optional columns are missing.
    case_cols = [
        "query_subject_id", "query_hadm_id", "query_stay_id", "query_row_id",
        "agree_top1", "retrieval_top_antibiotic", "classifier_top_antibiotic",
        "retrieval_top1_true", "classifier_top1_true",
        "retrieval_ranking", "classifier_ranking",
        "retrieval_top3_true_count", "classifier_top3_true_count",
        "top10_neighbor_subject_ids", "neighbor_similarities",
    ]

    # Keep only columns that actually exist in the dataframe.
    case_cols = [c for c in case_cols if c in work.columns]

    # Define output file paths.
    summary_path = OUTPUT_DIR / "agreement_disagreement_summary.csv"
    cases_path = OUTPUT_DIR / "agreement_disagreement_patient_cases.csv"

    # Save both the aggregate summary and the patient-level case table.
    summary.to_csv(summary_path, index=False)
    work[case_cols].to_csv(cases_path, index=False)

    # Print locations and a quick preview so the run is easy to check in the terminal.
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {cases_path}")
    print("\nAgreement/disagreement summary:")
    print(summary)

if __name__ == "__main__":
    main()
