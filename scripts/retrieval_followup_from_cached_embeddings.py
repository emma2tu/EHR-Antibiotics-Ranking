
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "antibiotics_labels.csv"
OUTPUT_DIR = BASE / "outputs"

RUN_NAME = "mean_embeddings_top10"
EMB_NPY = BASE / "cache" / "patient_paragraph_bioclinicalbert_mean_embeddings.npy"

FOLLOWUP_DIR = BASE / "followup_outputs" / RUN_NAME
PREDICTIONS_CSV = OUTPUT_DIR / "frozen_validation_patient_predictions.csv"

ANTIBIOTICS = [
    "CLINDAMYCIN", "ERYTHROMYCIN", "GENTAMICIN", "LEVOFLOXACIN",
    "OXACILLIN", "TETRACYCLINE", "TRIMETHOPRIM/SULFA", "VANCOMYCIN",
]
TEXT_COL = "patient_paragraph"
ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42
TOP_K = 10

def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return x / denom

def safe_int(value):
    if pd.isna(value):
        return None
    return int(value)

def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)

def recreate_split(df: pd.DataFrame):
    train_val, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def rank_antibiotics_for_one_patient(query_embedding, train_embeddings_norm, train_df, antibiotics, top_k=10):
    similarities = train_embeddings_norm @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_sims = similarities[top_indices]
    neighbors = train_df.iloc[top_indices].copy()

    sim_weights = np.maximum(top_sims, 0)
    if sim_weights.sum() <= 1e-12:
        sim_weights = np.ones_like(sim_weights) / len(sim_weights)
    else:
        sim_weights = sim_weights / sim_weights.sum()

    scores, fractions, weighted_scores = {}, {}, {}
    for ab in antibiotics:
        labels = neighbors[ab].astype(int).to_numpy()
        scores[ab] = int(labels.sum())
        fractions[ab] = float(labels.mean())
        weighted_scores[ab] = float(np.sum(labels * sim_weights))

    ranking = sorted(antibiotics, key=lambda ab: (scores[ab], weighted_scores[ab], fractions[ab], ab), reverse=True)

    neighbor_records = []
    for rank, (neighbor_pos, sim) in enumerate(zip(top_indices, top_sims), start=1):
        row = train_df.iloc[neighbor_pos]
        rec = {"neighbor_rank": rank, "similarity": safe_float(sim), "train_row_id": safe_int(row["row_id"])}
        for col in ID_COLS:
            if col in train_df.columns:
                rec[f"neighbor_{col}"] = safe_int(row[col])
        for ab in antibiotics:
            rec[f"{ab}_true"] = safe_int(row[ab])
        neighbor_records.append(rec)

    return {
        "retrieval_scores": scores,
        "retrieval_fractions": fractions,
        "retrieval_weighted_scores": weighted_scores,
        "retrieval_ranking": ranking,
        "top_neighbor_indices_in_train": top_indices.tolist(),
        "top_neighbor_similarities": [safe_float(x) for x in top_sims],
        "neighbors": neighbor_records,
    }

def main():
    FOLLOWUP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Run name: {RUN_NAME}")
    print(f"Embedding file: {EMB_NPY}")

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COL] + ANTIBIOTICS).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    train, val, test = recreate_split(df)
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    if not EMB_NPY.exists():
        raise FileNotFoundError(f"Embedding file not found: {EMB_NPY}")

    all_emb = np.load(EMB_NPY)
    if all_emb.shape[0] != len(df):
        raise ValueError(f"Embedding/data row mismatch: {all_emb.shape[0]} vs {len(df)}")

    X_train = all_emb[train["row_id"].to_numpy()]
    X_test = all_emb[test["row_id"].to_numpy()]
    X_train_norm = normalize_embeddings(X_train)
    X_test_norm = normalize_embeddings(X_test)

    classifier_pred_df = None
    if PREDICTIONS_CSV.exists():
        classifier_pred_df = pd.read_csv(PREDICTIONS_CSV)
        if "row_id" not in classifier_pred_df.columns:
            raise ValueError(f"{PREDICTIONS_CSV} missing row_id.")
        classifier_pred_df = classifier_pred_df.set_index("row_id", drop=False)
        print(f"Loaded classifier predictions: {PREDICTIONS_CSV}")

    summary_rows, detailed_results = [], []

    for i in tqdm(range(len(test)), desc=f"Retrieval ranking ({RUN_NAME})"):
        query_row = test.iloc[i]
        query_row_id = safe_int(query_row["row_id"])
        retrieval = rank_antibiotics_for_one_patient(X_test_norm[i], X_train_norm, train, ANTIBIOTICS, TOP_K)

        summary = {
            "query_row_id": query_row_id,
            "embedding_run_name": RUN_NAME,
            "retrieval_top_antibiotic": retrieval["retrieval_ranking"][0],
            "retrieval_ranking": " > ".join(retrieval["retrieval_ranking"]),
            "neighbor_similarities": json.dumps(retrieval["top_neighbor_similarities"]),
        }
        for col in ID_COLS:
            if col in test.columns:
                summary[f"query_{col}"] = safe_int(query_row[col])
        for col in ID_COLS:
            neighbor_key = f"neighbor_{col}"
            if retrieval["neighbors"] and neighbor_key in retrieval["neighbors"][0]:
                summary[f"top{TOP_K}_neighbor_{col}s"] = json.dumps([n[neighbor_key] for n in retrieval["neighbors"]])

        for ab in ANTIBIOTICS:
            summary[f"{ab}_retrieval_count"] = retrieval["retrieval_scores"][ab]
            summary[f"{ab}_retrieval_fraction"] = retrieval["retrieval_fractions"][ab]
            summary[f"{ab}_retrieval_weighted_score"] = retrieval["retrieval_weighted_scores"][ab]
            summary[f"{ab}_true"] = safe_int(query_row[ab])

        if classifier_pred_df is not None and query_row_id in classifier_pred_df.index:
            pred_row = classifier_pred_df.loc[query_row_id]
            if isinstance(pred_row, pd.DataFrame):
                pred_row = pred_row.iloc[0]
            for ab in ANTIBIOTICS:
                for suffix in ["proba", "pred", "threshold"]:
                    col_name = f"{ab}_{suffix}"
                    if col_name in pred_row.index:
                        summary[f"{ab}_classifier_{suffix}"] = safe_float(pred_row[col_name])

            available_proba = {ab: summary.get(f"{ab}_classifier_proba") for ab in ANTIBIOTICS if summary.get(f"{ab}_classifier_proba") is not None}
            if available_proba:
                classifier_ranking = sorted(available_proba.keys(), key=lambda ab: available_proba[ab], reverse=True)
                summary["classifier_top_antibiotic"] = classifier_ranking[0]
                summary["classifier_ranking"] = " > ".join(classifier_ranking)

        if "classifier_top_antibiotic" in summary:
            summary["top_antibiotic_agreement"] = summary["retrieval_top_antibiotic"] == summary["classifier_top_antibiotic"]

        summary_rows.append(summary)
        detailed_results.append({
            "query": {"row_id": query_row_id, **{col: safe_int(query_row[col]) for col in ID_COLS if col in test.columns}},
            "retrieval": retrieval,
            "classifier": {ab: {"proba": summary.get(f"{ab}_classifier_proba"), "pred": summary.get(f"{ab}_classifier_pred"), "threshold": summary.get(f"{ab}_classifier_threshold")} for ab in ANTIBIOTICS},
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = FOLLOWUP_DIR / f"combined_retrieval_classifier_summary_top{TOP_K}.csv"
    detailed_path = FOLLOWUP_DIR / f"combined_retrieval_classifier_detailed_top{TOP_K}.json"
    summary_df.to_csv(summary_path, index=False)
    with open(detailed_path, "w") as f:
        json.dump(detailed_results, f, indent=2)
    print("\nSaved follow-up outputs:")
    print(f"  Summary CSV:   {summary_path}")
    print(f"  Detailed JSON: {detailed_path}")

if __name__ == "__main__":
    main()
