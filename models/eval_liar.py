import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import confusion_matrix

MODEL_DIR = "models/roberta_text_v1"   # change if needed
MAX_LEN = 256

# LIAR label mapping (6-class -> binary)
# fake-ish: pants-fire, false, barely-true
# real-ish: half-true, mostly-true, true
FAKE_LABELS = {"pants-fire", "false", "barely-true"}
REAL_LABELS = {"half-true", "mostly-true", "true"}

def liar_to_binary(label: str):
    label = str(label).strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
        return 0
    return None  # drop anything unexpected

def load_liar_split(path: str):
    df = pd.read_csv(path, sep="\t", header=None)

    # Correct columns for YOUR LIAR file
    df = df.rename(columns={
        1: "liar_label",   # true / false
        13: "statement"    # text
    })

    # Map labels to binary
    df["label"] = df["liar_label"].astype(str).str.lower().map({
        "false": 1,
        "true": 0
    })

    # Drop invalid rows
    df["statement"] = df["statement"].astype(str)
    df = df.dropna(subset=["label", "statement"])

    # Match training format
    df["input_text"] = "TITLE:  BODY: " + df["statement"]

    print("After filtering:", df.shape)
    return df[["input_text", "label"]]

def predict_proba(model, inputs):
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
    return probs

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Use MPS if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Load + combine LIAR splits (or just test.tsv if you want)
    test_df = load_liar_split("LIAR/test.tsv")

    # Tokenize
    enc = tokenizer(
        test_df["input_text"].tolist(),
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    y_true = test_df["label"].to_numpy()
    p_fake = predict_proba(model, enc)
    y_pred = (p_fake >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, p_fake)

    print("LIAR (OOD) Evaluation")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall:    {rec:.4f}")
    print(f"f1:        {f1:.4f}")
    print(f"roc_auc:   {auc:.4f}")
    print(f"n:         {len(y_true)}")

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual REAL", "Actual FAKE"],
        columns=["Pred REAL", "Pred FAKE"]
    )
    print("\nConfusion Matrix:")
    print(cm_df)
    
    print("\nThreshold sweep (LIAR):")
    for t in [0.50, 0.60, 0.70, 0.80, 0.90]:
        y_pred_t = (p_fake >= t).astype(int)

        acc_t = accuracy_score(y_true, y_pred_t)
        prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
            y_true, y_pred_t, average="binary", zero_division=0
        )

        cm_t = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm_t.ravel()

        print(f"\nThreshold = {t:.2f}")
        print(f"acc={acc_t:.4f}  prec={prec_t:.4f}  rec={rec_t:.4f}  f1={f1_t:.4f}")
        print(f"TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()

