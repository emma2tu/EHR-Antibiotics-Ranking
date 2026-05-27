
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "antibiotics_labels.csv"
CACHE_DIR = BASE / "cache"
SAVE_PATH = CACHE_DIR / "patient_paragraph_bioclinicalbert_mean_embeddings.npy"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TEXT_COL = "patient_paragraph"
ANTIBIOTICS = [
    "CLINDAMYCIN", "ERYTHROMYCIN", "GENTAMICIN", "LEVOFLOXACIN",
    "OXACILLIN", "TETRACYCLINE", "TRIMETHOPRIM/SULFA", "VANCOMYCIN",
]
BATCH_SIZE = 32
SAVE_EVERY = 512

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / denom

def encode_mean_embeddings(texts: list[str], save_path: Path) -> np.ndarray:
    CACHE_DIR.mkdir(exist_ok=True)
    embeddings = []
    start_idx = 0

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
            raise ValueError(f"Cache has more rows than current data: {cached.shape[0]} vs {len(texts)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Encoding device: {device}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_start in tqdm(range(start_idx, len(texts), BATCH_SIZE), desc="Encoding BioClinicalBERT mean embeddings"):
            batch_texts = texts[batch_start: batch_start + BATCH_SIZE]
            enc = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            batch_embeddings = mean_pool(out.last_hidden_state, enc["attention_mask"])
            embeddings.extend(batch_embeddings.cpu().numpy())

            completed = len(embeddings)
            if completed % SAVE_EVERY == 0 or completed == len(texts):
                temp_path = Path(str(save_path) + ".tmp")
                with open(temp_path, "wb") as f:
                    np.save(f, np.vstack(embeddings))
                temp_path.replace(save_path)
                print(f"Saved mean embedding checkpoint: {completed}/{len(texts)} -> {save_path}")

    final = np.vstack(embeddings)
    print("Final mean embedding shape:", final.shape)
    return final

def main():
    print(f"Reading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TEXT_COL] + ANTIBIOTICS).reset_index(drop=True)
    print("Rows after dropping missing text/labels:", len(df))
    embeddings = encode_mean_embeddings(df[TEXT_COL].tolist(), SAVE_PATH)
    if embeddings.shape[0] != len(df):
        raise ValueError(f"Embedding/data mismatch: {embeddings.shape[0]} vs {len(df)}")
    print("Done.")
    print(f"Saved: {SAVE_PATH}")

if __name__ == "__main__":
    main()
