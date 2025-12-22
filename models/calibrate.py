import os
import json
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import log_loss

MODEL_DIR = "models/roberta_text_v1"
CALIB_TSV = "LIAR/valid.tsv"          # use LIAR valid for calibration (NOT test)
MAX_LEN = 256
OUT_FILE = os.path.join(MODEL_DIR, "temperature.json")

def load_liar_valid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)

    # Your LIAR: col 1 = label ("true"/"false"), col 13 = statement text
    df = df.rename(columns={1: "liar_label", 13: "statement"})
    df["label"] = df["liar_label"].astype(str).str.lower().map({"false": 1, "true": 0})

    df["statement"] = df["statement"].astype(str)
    df = df.dropna(subset=["label", "statement"]).copy()
    df["label"] = df["label"].astype(int)

    df["input_text"] = "TITLE:  BODY: " + df["statement"]
    return df[["input_text", "label"]]

def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    # logits: [N, 2]
    logits = logits - logits.max(axis=1, keepdims=True)  # stability
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)

def nll_with_temperature(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    probs = softmax_numpy(logits / T)
    return log_loss(y, probs, labels=[0, 1])

def grid_search_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    candidates = np.concatenate([
        np.linspace(0.5, 2.0, 16),
        np.linspace(2.0, 10.0, 17),
    ])
    best_T = 1.0
    best_nll = float("inf")

    for T in candidates:
        nll = nll_with_temperature(logits, y, float(T))
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)

    return best_T


def main():
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    # Load calibration data (LIAR valid)
    df = load_liar_valid(CALIB_TSV)
    texts = df["input_text"].tolist()
    y = df["label"].to_numpy()

    print(f"Calibration set size: {len(df)}")
    if len(df) == 0:
        raise ValueError("No calibration rows loaded. Check LIAR valid path/format.")

    # Tokenize
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Get logits
    with torch.no_grad():
        logits = model(**enc).logits.detach().cpu().numpy()  # shape [N,2]

    # Baseline NLL at T=1
    base_nll = nll_with_temperature(logits, y, T=1.0)

    # Find best T
    best_T = grid_search_temperature(logits, y)
    best_nll = nll_with_temperature(logits, y, T=best_T)

    print(f"Baseline (T=1.0) log-loss: {base_nll:.6f}")
    print(f"Best T: {best_T:.3f}")
    print(f"Calibrated log-loss: {best_nll:.6f}")

    # Save temperature.json
    payload = {
        "temperature": best_T,
        "calibration_dataset": CALIB_TSV,
        "max_len": MAX_LEN,
        "note": "Temperature scaling for RoBERTa logits. Apply logits/T before softmax.",
    }
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved calibration file to: {OUT_FILE}")

if __name__ == "__main__":
    main()
