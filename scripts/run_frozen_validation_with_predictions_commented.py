<<<<<<< HEAD
"""
Frozen BioClinicalBERT + LightGBM validation with patient-level predictions.

Purpose
-------
This script evaluates whether frozen BioClinicalBERT embeddings can support
antibiotic susceptibility/effectiveness prediction. It treats each antibiotic as
a separate binary classification task.

High-level workflow
-------------------
1. Load the labeled antibiotic dataset.
2. Drop rows missing either the patient text or any antibiotic labels.
3. Split patients into train / validation / test sets using a fixed random seed.
4. Encode each patient's text paragraph with BioClinicalBERT.
   - The BioClinicalBERT weights are frozen; no transformer fine-tuning happens.
   - Embeddings are cached to disk so long embedding runs can resume if interrupted.
5. Train one LightGBM classifier per antibiotic using the frozen embeddings.
6. Evaluate each antibiotic-specific classifier on the held-out test set.
7. Save:
   - Summary performance metrics
   - Patient-level predicted probabilities and binary predictions
   - JSON results
   - Pickled LightGBM models

Run from the repository root:
    python run_frozen_validation_with_predictions.py
"""
=======
# Frozen BioClinicalBERT + LightGBM validation with patient-level predictions.
# Treats each antibiotic as a separate binary classification task.
# BioClinicalBERT weights are frozen — no fine-tuning, just embedding extraction.
>>>>>>> ba4863a (update)

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


<<<<<<< HEAD
# =============================================================================
# Paths
# =============================================================================
# BASE is the folder containing this script. Defining all other paths relative to
# BASE makes the script easier to run from the repository root without hardcoding
# machine-specific paths.
BASE = Path(__file__).parent

# Input CSV containing one row per patient/stay and binary labels for each
# antibiotic.
DATA_PATH = BASE / "data" / "antibiotics_labels.csv"

# Cache folder stores expensive intermediate outputs, especially BERT embeddings.
CACHE_DIR = BASE / "cache"

# Output folder stores final metrics, predictions, and trained models.
OUTPUT_DIR = BASE / "outputs"

# This .npy cache stores the embedding matrix in the same row order as the
# cleaned dataframe after dropna() and reset_index(). Keeping this row order fixed
# is important because train/test splits later retrieve embeddings by row_id.
EMB_NPY = CACHE_DIR / "patient_paragraph_bioclinicalbert_embeddings.npy"


# =============================================================================
# Experiment settings
# =============================================================================
# Hugging Face model used to convert patient text into dense vector embeddings.
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Each antibiotic is modeled as its own binary classification task. For example,
# the CLINDAMYCIN classifier predicts CLINDAMYCIN_true from the text embedding.
=======
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent

DATA_PATH = BASE / "data" / "antibiotics_labels.csv"
CACHE_DIR = BASE / "cache"
OUTPUT_DIR = BASE / "outputs"

# Row order must match the cleaned dataframe after dropna() + reset_index().
EMB_NPY = CACHE_DIR / "patient_paragraph_bioclinicalbert_embeddings.npy"


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

>>>>>>> ba4863a (update)
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

<<<<<<< HEAD
# Column containing the already-constructed patient narrative / EHR text block.
TEXT_COL = "patient_paragraph"

# Identifier columns are copied into the prediction CSV so individual patient
# predictions can be traced back to the original dataset.
ID_COLS = ["subject_id", "hadm_id", "stay_id"]

# Split fractions. The validation set is created for consistency with the
# original runner, but this script does not use it for model selection.
TEST_SIZE = 0.10
VAL_SIZE = 0.10

# Fixed seed makes the train/validation/test split reproducible.
RANDOM_STATE = 42

# Number of bootstrap resamples used to estimate approximate 95% confidence
# intervals for AUROC, AUPRC, and F1.
N_BOOTSTRAPS = 200

# Pooling controls how token-level BioClinicalBERT outputs become one vector per
# patient. "cls" uses the first [CLS] token embedding, matching the original
# benchmark style. "mean" averages across real, non-padding tokens.
POOLING = "cls"

# Batch size for embedding generation. Larger batches may run faster on a GPU but
# require more memory.
ENCODE_BATCH_SIZE = 32

# Save embedding checkpoints every SAVE_EVERY patients so long encoding jobs can
# resume after interruptions.
SAVE_EVERY = 512

# LightGBM hyperparameters shared by all antibiotic-specific classifiers.
=======
TEXT_COL = "patient_paragraph"

# Copied into the prediction CSV so results can be traced back to patients.
ID_COLS = ["subject_id", "hadm_id", "stay_id"]

TEST_SIZE = 0.10
VAL_SIZE  = 0.10
RANDOM_STATE = 42

# Bootstraps for approximate 95% CIs on AUROC, AUPRC, F1.
N_BOOTSTRAPS = 200

# "cls" uses the [CLS] token; "mean" averages non-padding tokens.
POOLING = "cls"

ENCODE_BATCH_SIZE = 32
SAVE_EVERY = 512  # checkpoint interval (rows) during embedding

>>>>>>> ba4863a (update)
LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=30,
    random_state=RANDOM_STATE,
    verbose=-1,
    n_jobs=-1,
)


<<<<<<< HEAD
# =============================================================================
# BioClinicalBERT embedding helpers
# =============================================================================
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings while ignoring padding tokens.

    BioClinicalBERT returns one embedding per token. For retrieval or downstream
    classification, this script needs one fixed-length embedding per patient
    paragraph. Mean pooling averages only the real tokens and ignores padding.
    """

    # attention_mask has shape [batch_size, sequence_length].
    # Expanding it to [batch_size, sequence_length, hidden_size] lets us multiply
    # each token embedding by 1 for real tokens and 0 for padding tokens.
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Sum embeddings across the token dimension after padding tokens are zeroed.
    summed = torch.sum(last_hidden_state * mask, dim=1)

    # Count the number of real tokens in each sequence. Clamp avoids division by
    # zero in unusual cases where a sequence has no valid tokens.
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)

=======
# ---------------------------------------------------------------------------
# BioClinicalBERT embedding helpers
# ---------------------------------------------------------------------------
def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Average token embeddings, ignoring padding tokens.
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
>>>>>>> ba4863a (update)
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
<<<<<<< HEAD
    """Generate frozen BioClinicalBERT embeddings for a list of texts.

    Parameters
    ----------
    model_name:
        Hugging Face model name for the encoder.
    texts:
        List of patient text narratives to encode.
    save_path:
        Optional .npy file used for checkpointing/caching embeddings.
    batch_size:
        Number of texts encoded per forward pass.
    save_every:
        Number of completed embeddings between checkpoint saves.
    pooling:
        Either "cls" or "mean".
    force_restart:
        If True, ignore any existing cache and recompute embeddings.

    Returns
    -------
    np.ndarray
        Embedding matrix with shape [number_of_texts, hidden_size].
        For BioClinicalBERT, hidden_size is 768.
    """
=======
    # Returns embedding matrix [n_texts, hidden_size].
    # Resumes from disk if a partial cache exists.
>>>>>>> ba4863a (update)

    if pooling not in {"cls", "mean"}:
        raise ValueError("pooling must be either 'cls' or 'mean'")

    embeddings: list[np.ndarray] = []
    start_idx = 0

<<<<<<< HEAD
    # If a cache already exists, either load it completely or resume from the
    # number of rows already saved. This is especially useful because embedding
    # all patient paragraphs can take a long time.
    if save_path is not None and save_path.exists() and not force_restart:
        cached = np.load(save_path)

        # Complete cache: return immediately without running BioClinicalBERT.
=======
    if save_path is not None and save_path.exists() and not force_restart:
        cached = np.load(save_path)

>>>>>>> ba4863a (update)
        if cached.shape[0] == len(texts):
            print(f"Loading complete cached embeddings from {save_path}")
            return cached

<<<<<<< HEAD
        # Partial cache: continue encoding from the first unfinished row.
=======
>>>>>>> ba4863a (update)
        if cached.shape[0] < len(texts):
            embeddings = list(cached)
            start_idx = cached.shape[0]
            print(f"Resuming embeddings from {save_path}: {start_idx}/{len(texts)} completed")
        else:
<<<<<<< HEAD
            # More cached rows than current data usually means the cleaned input
            # dataset changed. Reusing this cache would misalign rows.
=======
            # More cached rows than data — likely a dataset change, so row alignment is off.
>>>>>>> ba4863a (update)
            raise ValueError(
                f"Embedding cache has more rows than current data: {cached.shape[0]} vs {len(texts)}. "
                "Delete the cache and rerun."
            )

<<<<<<< HEAD
    # Load tokenizer and encoder model from Hugging Face.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Use GPU if available; otherwise fall back to CPU.
=======
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

>>>>>>> ba4863a (update)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoding device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model.to(device)
<<<<<<< HEAD
    model.eval()  # evaluation mode disables dropout for deterministic embeddings

    # no_grad() avoids storing gradients because this script only extracts frozen
    # embeddings and does not fine-tune BioClinicalBERT.
=======
    model.eval()  # dropout off for deterministic output

>>>>>>> ba4863a (update)
    with torch.no_grad():
        for batch_start in tqdm(
            range(start_idx, len(texts), batch_size),
            desc=f"Encoding BioClinicalBERT ({pooling})",
        ):
            batch_texts = texts[batch_start : batch_start + batch_size]

<<<<<<< HEAD
            # Tokenize the text batch. max_length=512 is the standard BERT limit;
            # longer patient paragraphs are truncated.
=======
            # max_length=512 is the BERT limit; longer paragraphs are truncated.
>>>>>>> ba4863a (update)
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            enc = {k: v.to(device) for k, v in enc.items()}

<<<<<<< HEAD
            # Forward pass through BioClinicalBERT. last_hidden_state has shape:
            # [batch_size, sequence_length, hidden_size].
            out = model(**enc)
            last_hidden = out.last_hidden_state

            # Convert token-level representations into one vector per patient.
            if pooling == "cls":
                # [CLS] embedding: first token representation.
                batch_embeddings = last_hidden[:, 0, :]
            else:
                # Mean embedding: average across non-padding tokens.
=======
            out = model(**enc)
            last_hidden = out.last_hidden_state

            if pooling == "cls":
                batch_embeddings = last_hidden[:, 0, :]
            else:
>>>>>>> ba4863a (update)
                batch_embeddings = mean_pool(last_hidden, enc["attention_mask"])

            embeddings.extend(batch_embeddings.cpu().numpy())

            completed = len(embeddings)

<<<<<<< HEAD
            # Periodically save a temporary checkpoint, then atomically replace
            # the target cache file. This helps avoid corrupting the cache if the
            # process is interrupted during a save.
=======
            # Atomic checkpoint: write to .tmp then rename to avoid partial writes.
>>>>>>> ba4863a (update)
            if save_path is not None and (completed % save_every == 0 or completed == len(texts)):
                save_path.parent.mkdir(exist_ok=True)
                temp_path = Path(str(save_path) + ".tmp")
                with open(temp_path, "wb") as f:
                    np.save(f, np.vstack(embeddings))
                temp_path.replace(save_path)
                print(f"Saved embedding checkpoint: {completed}/{len(texts)} -> {save_path}")

    return np.vstack(embeddings)


<<<<<<< HEAD
# =============================================================================
# Model training and evaluation
# =============================================================================
def evaluate(X_tr, X_te, y_tr, y_te, n_bootstraps=N_BOOTSTRAPS):
    """Train and evaluate one antibiotic-specific LightGBM classifier.

    This function is called once per antibiotic. It trains on the training
    embeddings/labels, predicts probabilities on the test embeddings, chooses an
    F1-maximizing threshold, and returns both summary metrics and patient-level
    predictions.
    """

    # Train a binary LightGBM classifier for one antibiotic.
    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr)

    # Probability that each test patient belongs to the positive class.
    proba = clf.predict_proba(X_te)[:, 1]

    # Threshold-independent ranking metrics.
    auroc = float(roc_auc_score(y_te, proba))
    auprc = float(average_precision_score(y_te, proba))

    # Precision-recall curve is used to find the threshold that maximizes F1.
    precision, recall, thresholds = precision_recall_curve(y_te, proba)

    if len(thresholds):
        # sklearn returns one fewer threshold than precision/recall values, so
        # precision[:-1] and recall[:-1] are aligned with thresholds.
=======
# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------
def evaluate(X_tr, X_te, y_tr, y_te, n_bootstraps=N_BOOTSTRAPS):
    # Train one LightGBM classifier and evaluate it on the test set.
    # Returns summary metrics, patient-level predictions, and the fitted model.

    clf = LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]

    auroc = float(roc_auc_score(y_te, proba))
    auprc = float(average_precision_score(y_te, proba))

    precision, recall, thresholds = precision_recall_curve(y_te, proba)

    if len(thresholds):
        # sklearn returns one fewer threshold than precision/recall values.
>>>>>>> ba4863a (update)
        f1s = 2 * precision[:-1] * recall[:-1] / np.maximum(
            precision[:-1] + recall[:-1],
            1e-10,
        )
        best = int(np.argmax(f1s))
        opt_thr = float(thresholds[best])
        f1 = float(f1s[best])
<<<<<<< HEAD

        # Convert probabilities into binary predictions using the selected
        # antibiotic-specific threshold.
        pred = (proba >= opt_thr).astype(int)

        # MCC summarizes binary classification quality while accounting for all
        # four confusion matrix cells and class imbalance.
        mcc = float(matthews_corrcoef(y_te, pred))
    else:
        # Fallback for edge cases with no usable threshold.
=======
        pred = (proba >= opt_thr).astype(int)
        mcc = float(matthews_corrcoef(y_te, pred))
    else:
>>>>>>> ba4863a (update)
        opt_thr = 0.5
        pred = (proba >= opt_thr).astype(int)
        f1 = 0.0
        mcc = 0.0

<<<<<<< HEAD
    # Basic class balance information for the held-out test set.
=======
>>>>>>> ba4863a (update)
    n_pos = int(y_te.sum())
    n_neg = int(len(y_te) - n_pos)
    prev = float(n_pos / len(y_te))

<<<<<<< HEAD
    # Bootstrap confidence intervals. Each bootstrap sample resamples test
    # patients with replacement and recomputes metrics.
=======
    # Bootstrap CIs via resampling test patients with replacement.
>>>>>>> ba4863a (update)
    roc_boots, prc_boots, f1_boots = [], [], []

    y_te_arr = np.asarray(y_te)
    for _ in range(n_bootstraps):
        idx = resample(np.arange(len(y_te_arr)), replace=True, random_state=None)
        yt = y_te_arr[idx]
        yp = proba[idx]

<<<<<<< HEAD
        # AUROC requires both positive and negative examples. Skip bootstrap
        # samples that accidentally contain only one class.
=======
        # AUROC is undefined if the bootstrap sample has only one class.
>>>>>>> ba4863a (update)
        if len(np.unique(yt)) < 2:
            continue

        roc_boots.append(float(roc_auc_score(yt, yp)))
        prc_boots.append(float(average_precision_score(yt, yp)))

<<<<<<< HEAD
        # Recompute the best possible F1 on the bootstrap sample.
=======
>>>>>>> ba4863a (update)
        pr_, rc_, th_ = precision_recall_curve(yt, yp)
        if len(th_):
            f1b = 2 * pr_[:-1] * rc_[:-1] / np.maximum(pr_[:-1] + rc_[:-1], 1e-10)
            f1_boots.append(float(np.max(f1b)))
        else:
            f1_boots.append(0.0)

    def ci95(arr):
<<<<<<< HEAD
        """Return bootstrap mean and percentile-based 95% confidence interval."""

=======
        # Percentile-based 95% CI from bootstrap samples.
>>>>>>> ba4863a (update)
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

<<<<<<< HEAD
    # Store all summary metrics in one dictionary so they can be written to CSV
    # and JSON later.
=======
>>>>>>> ba4863a (update)
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

<<<<<<< HEAD
    # Patient-level outputs are kept separately from summary metrics so they can
    # be saved in a prediction table.
=======
>>>>>>> ba4863a (update)
    predictions = {
        "true": y_te_arr.astype(int),
        "proba": proba.astype(float),
        "pred": pred.astype(int),
    }

    return metrics, predictions, clf


<<<<<<< HEAD
# =============================================================================
# Main script
# =============================================================================
def main():
    # Create cache/output folders if they do not already exist.
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Load and clean data
    # -------------------------------------------------------------------------
    print(f"Reading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Keep only rows with usable patient text and complete labels for all eight
    # antibiotics. This ensures every classifier is trained/evaluated on the same
    # patient population.
    df = df.dropna(subset=[TEXT_COL] + ANTIBIOTICS).reset_index(drop=True)

    # row_id preserves the cleaned dataframe row order. This is essential because
    # embeddings are generated for the full dataframe first, then train/test rows
    # retrieve their corresponding embeddings by row_id.
=======
# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
def main():
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # --- Load and clean data ---
    print(f"Reading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Drop rows missing text or any antibiotic label so all classifiers share the same patient set.
    df = df.dropna(subset=[TEXT_COL] + ANTIBIOTICS).reset_index(drop=True)

    # row_id tracks position in the cleaned df; used to pull embeddings after splitting.
>>>>>>> ba4863a (update)
    df["row_id"] = np.arange(len(df))

    print(f"Rows after dropping missing text/labels: {len(df)}")

<<<<<<< HEAD
    # Some datasets may be missing one or more identifier columns. Rather than
    # failing, the script keeps whichever identifiers are available.
=======
>>>>>>> ba4863a (update)
    missing_ids = [col for col in ID_COLS if col not in df.columns]
    if missing_ids:
        print(f"Note: these ID columns were not found and will be skipped: {missing_ids}")

    available_id_cols = [col for col in ID_COLS if col in df.columns]

<<<<<<< HEAD
    # -------------------------------------------------------------------------
    # Reproducible train / validation / test split
    # -------------------------------------------------------------------------
    # First hold out the final test set.
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Then split the remaining data into train and validation. The validation set
    # is printed and preserved for consistency, but this script does not tune
    # hyperparameters on it.
    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(f"Train: {len(train)}  Val (unused): {len(val)}  Test: {len(test)}")

    # -------------------------------------------------------------------------
    # Generate or load frozen BioClinicalBERT embeddings
    # -------------------------------------------------------------------------
    if EMB_NPY.exists():
        print(f"Found embedding cache: {EMB_NPY}")

    # The embedding matrix is generated for all cleaned rows in dataframe order.
    # all_emb[i] corresponds to df row i.
=======
    # --- Train / val / test split ---
    train_val, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Val set is kept for consistency but not used for hyperparameter tuning here.
    train, val = train_test_split(train_val, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    print(f"Train: {len(train)}  Val (unused): {len(val)}  Test: {len(test)}")

    # --- Generate or load frozen BioClinicalBERT embeddings ---
    if EMB_NPY.exists():
        print(f"Found embedding cache: {EMB_NPY}")

    # all_emb[i] corresponds to df row i (cleaned dataframe order).
>>>>>>> ba4863a (update)
    all_emb = encode_texts(
        MODEL_NAME,
        df[TEXT_COL].tolist(),
        save_path=EMB_NPY,
        batch_size=ENCODE_BATCH_SIZE,
        save_every=SAVE_EVERY,
        pooling=POOLING,
    )

<<<<<<< HEAD
    # Safety check: prevents silent row misalignment if the cache came from a
    # different version of the dataset.
=======
>>>>>>> ba4863a (update)
    if all_emb.shape[0] != len(df):
        raise ValueError(
            f"Embedding/data row mismatch: embeddings={all_emb.shape[0]}, df={len(df)}. "
            "Delete cache and rerun."
        )

<<<<<<< HEAD
    # Pull train/test embeddings using row_id so they match the rows selected by
    # train_test_split.
    X_train = all_emb[train["row_id"].to_numpy()]
    X_test = all_emb[test["row_id"].to_numpy()]

    print(f"\nEmbedding shapes — Train: {X_train.shape}  Test: {X_test.shape}\n")

    # Dictionaries collect full JSON-style results and fitted models.
    results = {}
    models = {}

    # Start patient-level prediction table with identifiers and row_id.
    pred_df = test[available_id_cols + ["row_id"]].copy()

    # Include true labels for each antibiotic before adding probabilities and
    # binary predictions.
=======
    X_train = all_emb[train["row_id"].to_numpy()]
    X_test  = all_emb[test["row_id"].to_numpy()]

    print(f"\nEmbedding shapes — Train: {X_train.shape}  Test: {X_test.shape}\n")

    results = {}
    models  = {}

    pred_df = test[available_id_cols + ["row_id"]].copy()

    # Ground-truth labels first, then probabilities/predictions below.
>>>>>>> ba4863a (update)
    for ab in ANTIBIOTICS:
        pred_df[f"{ab}_true"] = test[ab].astype(int).to_numpy()

    metric_rows = []

<<<<<<< HEAD
    # Console table header for quick progress monitoring.
    print(f"{'Antibiotic':<22}  {'AUROC':>6}  {'95% CI':>16}  {'AUPRC':>6}  {'F1':>6}  {'MCC':>6}  {'Prev':>5}")
    print("-" * 82)

    # -------------------------------------------------------------------------
    # Train and evaluate one model per antibiotic
    # -------------------------------------------------------------------------
    for ab in ANTIBIOTICS:
        # Binary labels for this antibiotic.
=======
    print(f"{'Antibiotic':<22}  {'AUROC':>6}  {'95% CI':>16}  {'AUPRC':>6}  {'F1':>6}  {'MCC':>6}  {'Prev':>5}")
    print("-" * 82)

    # --- Train and evaluate one model per antibiotic ---
    for ab in ANTIBIOTICS:
>>>>>>> ba4863a (update)
        y_tr = train[ab].astype(int).reset_index(drop=True)
        y_te = test[ab].astype(int).reset_index(drop=True)

        metrics, predictions, clf = evaluate(X_train, X_test, y_tr, y_te)

<<<<<<< HEAD
        # JSON-friendly result structure.
        results[ab] = {
            "metrics": metrics,
            "predictions": {
                "true": predictions["true"].tolist(),
                "proba": predictions["proba"].tolist(),
                "pred": predictions["pred"].tolist(),
            },
        }

        # Save fitted classifier in memory so all antibiotic models can be
        # pickled together at the end.
        models[ab] = clf

        # Add patient-level predictions to the prediction CSV.
        pred_df[f"{ab}_proba"] = predictions["proba"]
        pred_df[f"{ab}_pred"] = predictions["pred"]
        pred_df[f"{ab}_threshold"] = metrics["optimal_threshold"]

        # One row per antibiotic in the summary metrics CSV.
        metric_rows.append({"antibiotic": ab, **metrics})

        # Print compact performance summary for this antibiotic.
=======
        results[ab] = {
            "metrics": metrics,
            "predictions": {
                "true":  predictions["true"].tolist(),
                "proba": predictions["proba"].tolist(),
                "pred":  predictions["pred"].tolist(),
            },
        }

        models[ab] = clf

        pred_df[f"{ab}_proba"]     = predictions["proba"]
        pred_df[f"{ab}_pred"]      = predictions["pred"]
        pred_df[f"{ab}_threshold"] = metrics["optimal_threshold"]

        metric_rows.append({"antibiotic": ab, **metrics})

>>>>>>> ba4863a (update)
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

<<<<<<< HEAD
    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    metrics_path = OUTPUT_DIR / "frozen_validation_metrics_comprehensive.csv"
    pred_path = OUTPUT_DIR / "frozen_validation_patient_predictions.csv"
    json_path = OUTPUT_DIR / "frozen_validation_results_with_predictions.json"
    models_path = OUTPUT_DIR / "frozen_validation_lightgbm_models.pkl"

    # Human-readable tables.
    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    # Full nested results, including patient-level arrays.
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Trained LightGBM classifiers for possible reuse without retraining.
=======
    # --- Save outputs ---
    metrics_path = OUTPUT_DIR / "frozen_validation_metrics_comprehensive.csv"
    pred_path    = OUTPUT_DIR / "frozen_validation_patient_predictions.csv"
    json_path    = OUTPUT_DIR / "frozen_validation_results_with_predictions.json"
    models_path  = OUTPUT_DIR / "frozen_validation_lightgbm_models.pkl"

    metrics_df.to_csv(metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

>>>>>>> ba4863a (update)
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
