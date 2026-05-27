# Generate mean-pooled BioClinicalBERT embeddings for the antibiotic-ranking
# retrieval pipeline.
#
# This script is separate from the classifier validation script because retrieval
# works better with one fixed patient-level vector per text. Here, each patient's
# BioClinicalBERT token embeddings are averaged across real, non-padding tokens
# to create a mean-pooled embedding.

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# Project paths are defined relative to this script so the script can be run
# from the repository without hard-coding absolute file locations.
BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "antibiotics_labels.csv"
CACHE_DIR = BASE / "cache"
SAVE_PATH = CACHE_DIR / "patient_paragraph_bioclinicalbert_mean_embeddings.npy"

# Hugging Face model used to encode each patient paragraph.
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Input text column and label columns required for the retrieval dataset.
# Rows missing either text or any antibiotic label are dropped in main().
TEXT_COL = "patient_paragraph"
ANTIBIOTICS = [
    "CLINDAMYCIN", "ERYTHROMYCIN", "GENTAMICIN", "LEVOFLOXACIN",
    "OXACILLIN", "TETRACYCLINE", "TRIMETHOPRIM/SULFA", "VANCOMYCIN",
]

# Batch size controls memory usage during model inference.
# SAVE_EVERY controls how often partial embeddings are checkpointed to disk.
BATCH_SIZE = 32
SAVE_EVERY = 512

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # Expand the attention mask so it can be multiplied against every hidden
    # dimension for every token.
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    # Sum only the embeddings for real tokens. Padding tokens are multiplied by 0.
    summed = torch.sum(last_hidden_state * mask, dim=1)

    # Count the number of real tokens per patient paragraph.
    # The clamp prevents division by zero if an unexpected empty input appears.
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)

    # Return one mean-pooled vector per patient paragraph.
    return summed / denom

def encode_mean_embeddings(texts: list[str], save_path: Path) -> np.ndarray:
    # Ensure the cache folder exists before saving checkpoints.
    CACHE_DIR.mkdir(exist_ok=True)

    # embeddings stores completed batches. start_idx allows interrupted runs to
    # resume from the last saved checkpoint instead of restarting from scratch.
    embeddings = []
    start_idx = 0

    # Reuse the saved .npy file when possible. This is important because encoding
    # all patient paragraphs can take a long time.
    if save_path.exists():
        cached = np.load(save_path)
        if cached.shape[0] == len(texts):
            print(f"Complete mean embedding cache already exists: {save_path}")
            print("Shape:", cached.shape)
            return cached
        if cached.shape[0] < len(texts):
            embeddings = list(cached)
            start_idx = cached.shape[0]
            print(f"Resuming mean embeddings: {start_idx}/{len(texts)} completed")
        else:
            # A larger cache means the cached embeddings no longer match the
            # current filtered dataset, so continuing would misalign rows.
            raise ValueError(f"Cache has more rows than current data: {cached.shape[0]} vs {len(texts)}")

    # Load the tokenizer and frozen BioClinicalBERT model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    # Use GPU acceleration when available; otherwise, fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoding device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Move model to the selected device and set evaluation mode because this
    # script only performs inference, not fine-tuning.
    model.to(device)
    model.eval()

    # Disable gradient tracking to reduce memory use and speed up inference.
    with torch.no_grad():
        for batch_start in tqdm(range(start_idx, len(texts), BATCH_SIZE), desc="Encoding BioClinicalBERT mean embeddings"):
            # Select the current batch of patient paragraphs.
            batch_texts = texts[batch_start: batch_start + BATCH_SIZE]

            # Tokenize text into BERT inputs. Text is truncated to the standard
            # 512-token BERT limit and padded within each batch.
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}

            # Run BioClinicalBERT and mean-pool the final hidden states into one
            # fixed-length embedding per patient paragraph.
            out = model(**enc)
            batch_embeddings = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeddings.extend(batch_embeddings.cpu().numpy())

            # Periodically save a checkpoint so long embedding runs can resume if
            # interrupted.
            completed = len(embeddings)
            if completed % SAVE_EVERY == 0 or completed == len(texts):
                temp_path = Path(str(save_path) + ".tmp")
                with open(temp_path, "wb") as f:
                    np.save(f, np.vstack(embeddings))
                temp_path.replace(save_path)
                print(f"Saved mean embedding checkpoint: {completed}/{len(texts)} -> {save_path}")

    # Stack the list of per-patient vectors into one 2D array:
    # rows = patients, columns = BioClinicalBERT embedding dimensions.
    final = np.vstack(embeddings)
    print("Final mean embedding shape:", final.shape)
    return final

def main():
    # Load the labeled antibiotic dataset.
    print(f"Reading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Keep only complete rows so every saved embedding corresponds to a patient
    # with text and all eight antibiotic labels.
    df = df.dropna(subset=[TEXT_COL] + ANTIBIOTICS).reset_index(drop=True)
    print("Rows after dropping missing text/labels:", len(df))

    # Encode every patient paragraph and save/load the cached embedding matrix.
    embeddings = encode_mean_embeddings(df[TEXT_COL].tolist(), SAVE_PATH)

    # Final safety check: one embedding row should match one filtered data row.
    if embeddings.shape[0] != len(df):
        raise ValueError(f"Embedding/data mismatch: {embeddings.shape[0]} vs {len(df)}")
    print("Done.")
    print(f"Saved: {SAVE_PATH}")

if __name__ == "__main__":
    main()
