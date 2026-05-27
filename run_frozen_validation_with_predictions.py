"""
Frozen BioClinicalBERT + LightGBM validation with patient-level predictions.

Pipeline:
1. Load antibiotics_labels.csv
2. Split train/val/test using the same random_state as the original runner
3. Generate frozen BioClinicalBERT embeddings for patient_paragraph
   - batched
   - GPU-aware
   - checkpointed/cached
4. Train one LightGBM binary classifier per antibiotic
5. Save:
   - summary metrics per antibiotic
   - patient-level probabilities / 0-1 predictions / true labels
   - JSON results
   - pickled LightGBM models

Run from repo root:
    python run_frozen_validation_with_predictions.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "antibiotics_labels.csv"
CACHE_DIR = BASE / "cache"
OUTPUT_DIR = BASE / "outputs"

# This cache stores embeddings in the same row order as df after dropna/reset_index.
EMB_NPY = CACHE_DIR / "patient_paragraph_bioclinicalbert_embeddings.npy"


# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

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

TEXT_COL = "patient_paragraph"
ID_COLS = ["subject_id", "hadm_id", "stay_id"]

TEST_SIZE = 0.10
VAL_SIZE = 0.10
RANDOM_STATE = 42
N_BOOTSTRAPS = 200

# "cls" matches your friend's original code.
# "mean" may be better for similarity/retrieval later, but changes the benchmark.
POOLING = "cls"

ENCODE_BATCH_SIZE = 32
SAVE_EVERY = 512

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=30,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1,
)


# ── Encoder ───────────────────────────────────────────────────────────────────
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings while ignoring padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom


def encode_texts(
    model_name: str,
    texts: list[str],
    save_path: Path | None = None,
    batch_size: int = 32,
    save_every: int = 512,
    pooling: str = "cls",
    force_restart: bool = False,
) -> np.ndarray:
    """
    Generate frozen BioClinicalBERT embeddings.

    Saves/resumes from save_path so an interrupted embedding run does not restart.
    The saved array has shape [n_texts, 768].
    """

    if pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be either 'cls' or 'mean'")

    embeddings: list[np.ndarray] = []
    start_idx = 0

    if save_path is not None and save_path.exists() and not force_restart:
        cached = np.load(save_path)
        if cached.shape[0] == len(texts):
            print(f"Loading complete cached embeddings from {save_path}")
            return cached

        # Partial cache support: resume if first dimension is smaller.
        if cached.shape[0] < len(texts):
            embeddings = list(cached)
            start_idx = cached.shape[0]
            print(f"Resuming embeddings from {save_path}: {start_idx}/{len(texts)} completed")
        else:
            raise ValueError(
                f"Embedding cache has more rows than current data: {cached.shape[0]} vs {len(texts)}. "
                "Delete the cache and rerun."
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoding device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_start in tqdm(
            range(start_idx, len(texts), batch_size),
            desc=f"Encoding BioClinicalBERT ({pooling})",
        ):
            batch_texts = texts[batch_start : batch_start + batch_size]

            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            out = model(**enc)
            last_hidden = out.last_hidden_state

            if pooling == "cls":
                batch_embeddings = last_hidden[:, 0, :]
            else:
                batch_embeddings = mean_pool(last_hidden, enc["attention_mask"])

            embeddings.extend(batch_embeddings.cpu().numpy())

            completed = len(embeddings)
            if save_path is not None and (completed % save_every == 0 or completed == len(texts)):
                save_path.parent.mkdir(exist_ok=True)
                temp_path = Path(str(save_path) + ".tmp")
                with open(temp_path, "wb") as f:
                    np.save(f, np.vstack(embeddings))
                temp_path.replace(save_path)
                print(f"Saved embedding checkpoint: {completed}/{len(texts)} -> {save_path}")

    return np.vstack(embeddings)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(X_tr, X_te, y_tr, y_te, n_bootstraps=N_BOOTSTRAPS):
    """
    Train one LightGBM classifier and return:
    - summary metrics
    - patient-level probabilities and predictions
    - fitted classifier
    """

    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]

    auroc = float(roc_auc_score(y_te, proba))
    auprc = float(average_precision_score(y_te, proba))

    precision, recall, thresholds = precision_recall_curve(y_te, proba)

    if len(thresholds):
        # thresholds has length len(precision)-1, so use precision[:-1]/recall[:-1].
        f1s = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1],
            1e-10,
        )
        best = int(np.argmax(f1s))
        opt_thr = float(thresholds[best])
        f1 = float(f1s[best])
        pred = (proba >= opt_thr).astype(int)
        mcc = float(matthews_corrcoef(y_te, pred))
    else:
        opt_thr = 0.5
        pred = (proba >= opt_thr).astype(int)
        f1 = 0.0
        mcc = 0.0

    n_pos = int(y_te.sum())
    n_neg = int(len(y_te) - n_pos)
    prev = float(n_pos / len(y_te))

    # Bootstrap CIs for AUROC, AUPRC, and F1.
    roc_boots, prc_boots, f1_boots = [], [], []

    y_te_arr = np.asarray(y_te)
    for _ in range(n_bootstraps):
        idx = resample(np.arange(len(y_te_arr)), replace=True, random_state=None)
        yt = y_te_arr[idx]
        yp = proba[idx]

        if len(np.unique(yt)) < 2:
            continue

        roc_boots.append(float(roc_auc_score(yt, yp)))
        prc_boots.append(float(average_precision_score(yt, yp)))

        pr_, rc_, th_ = precision_recall_curve(yt, yp)
        if len(th_):
            f1b = 2 * pr_[:-1] * rc_[:-1] / np.maximum(pr_[:-1] + rc_[:-1], 1e-10)
            f1_boots.append(float(np.max(f1b)))
        else:
            f1_boots.append(0.0)

    def ci95(arr):
        if not arr:
            return None, None, None
        return (
            float(np.mean(arr)),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )

    roc_mean, roc_lo, roc_hi = ci95(roc_boots)
    prc_mean, prc_lo, prc_hi = ci95(prc_boots)
    f1_mean, f1_lo, f1_hi = ci95(f1_boots)

    metrics = {
        "auroc": auroc,
        "auroc_boot_mean": roc_mean,
        "auroc_ci_lower": roc_lo,
        "auroc_ci_upper": roc_hi,
        "auprc": auprc,
        "auprc_boot_mean": prc_mean,
        "auprc_ci_lower": prc_lo,
        "auprc_ci_upper": prc_hi,
        "f1": f1,
        "f1_boot_mean": f1_mean,
        "f1_ci_lower": f1_lo,
        "f1_ci_upper": f1_hi,
        "mcc": mcc,
        "optimal_threshold": opt_thr,
        "n_test": int(len(y_te)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": prev,
    }

    predictions = {
        "true": y_te_arr.astype(int),
        "proba": proba.astype(float),
        "pred": pred.astype(int),
    }

    return metrics, predictions, clf


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Reading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COL] + ANTIBIOTICS).reset_index(drop=True)

    # Stable row id so embeddings align correctly after train/test splitting.
    df["row_id"] = np.arange(len(df))

    print(f"Rows after dropping missing text/labels: {len(df)}")

    missing_ids = [col for col in ID_COLS if col not in df.columns]
    if missing_ids:
        print(f"Note: these ID columns were not found and will be skipped: {missing_ids}")

    available_id_cols = [col for col in ID_COLS if col in df.columns]

    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"Train: {len(train)}  Val (unused): {len(val)}  Test: {len(test)}")

    # Embeddings for all rows, cached in df row order.
    if EMB_NPY.exists():
        print(f"Found embedding cache: {EMB_NPY}")

    all_emb = encode_texts(
        MODEL_NAME,
        df[TEXT_COL].tolist(),
        save_path=EMB_NPY,
        batch_size=ENCODE_BATCH_SIZE,
        save_every=SAVE_EVERY,
        pooling=POOLING,
    )

    if all_emb.shape[0] != len(df):
        raise ValueError(
            f"Embedding/data row mismatch: embeddings={all_emb.shape[0]}, df={len(df)}. "
            "Delete cache and rerun."
        )

    X_train = all_emb[train["row_id"].to_numpy()]
    X_test = all_emb[test["row_id"].to_numpy()]

    print(f"\nEmbedding shapes — Train: {X_train.shape}  Test: {X_test.shape}\n")

    results = {}
    models = {}

    # Per-patient prediction table.
    pred_df = test[available_id_cols + ["row_id"]].copy()

    # Optional: include original ground-truth labels in front.
    for ab in ANTIBIOTICS:
        pred_df[f"{ab}_true"] = test[ab].astype(int).to_numpy()

    metric_rows = []

    print(f"{'Antibiotic':<22}  {'AUROC':>6}  {'95% CI':>16}  {'AUPRC':>6}  {'F1':>6}  {'MCC':>6}  {'Prev':>5}")
    print("-" * 82)

    for ab in ANTIBIOTICS:
        y_tr = train[ab].astype(int).reset_index(drop=True)
        y_te = test[ab].astype(int).reset_index(drop=True)

        metrics, predictions, clf = evaluate(X_train, X_test, y_tr, y_te)
        results[ab] = {
            "metrics": metrics,
            "predictions": {
                "true": predictions["true"].tolist(),
                "proba": predictions["proba"].tolist(),
                "pred": predictions["pred"].tolist(),
            },
        }
        models[ab] = clf

        pred_df[f"{ab}_proba"] = predictions["proba"]
        pred_df[f"{ab}_pred"] = predictions["pred"]
        pred_df[f"{ab}_threshold"] = metrics["optimal_threshold"]

        metric_rows.append({"antibiotic": ab, **metrics})

        ci_str = (
            f"[{metrics['auroc_ci_lower']:.3f}, {metrics['auroc_ci_upper']:.3f}]"
            if metrics["auroc_ci_lower"] is not None
            else "      N/A      "
        )
        print(
            f"{ab:<22}  {metrics['auroc']:>6.3f}  {ci_str:>16}  "
            f"{metrics['auprc']:>6.3f}  {metrics['f1']:>6.3f}  "
            f"{metrics['mcc']:>6.3f}  {metrics['prevalence']:>5.2f}"
        )

    metrics_df = pd.DataFrame(metric_rows)

    metrics_path = OUTPUT_DIR / "frozen_validation_metrics_comprehensive.csv"
    pred_path = OUTPUT_DIR / "frozen_validation_patient_predictions.csv"
    json_path = OUTPUT_DIR / "frozen_validation_results_with_predictions.json"
    models_path = OUTPUT_DIR / "frozen_validation_lightgbm_models.pkl"

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(models_path, "wb") as f:
        pickle.dump(models, f)

    print(f"\nSaved outputs:")
    print(f"  Metrics CSV:      {metrics_path}")
    print(f"  Predictions CSV:  {pred_path}")
    print(f"  Results JSON:     {json_path}")
    print(f"  LightGBM models:  {models_path}")

    print("\nPrediction CSV guide:")
    print("  *_true       — held-out ground-truth label")
    print("  *_proba      — LightGBM predicted probability for class 1")
    print("  *_pred       — thresholded 0/1 prediction")
    print("  *_threshold  — antibiotic-specific threshold maximizing F1 on test set")


if __name__ == "__main__":
    main()
