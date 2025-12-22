import os
import pandas as pd

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "eval_outputs_fulltext_V2", "predictions.csv")

# If your CSV is elsewhere, change CSV_PATH accordingly.


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes actual/predicted labels into:
      actual_y:  REAL or FAKE
      pred_y:    REAL or FAKE

    Mapping assumptions:
      actual_bucket == 'real'   -> actual_y = REAL
      actual_bucket == 'misinfo'-> actual_y = FAKE
    """
    df = df.copy()

    # Predicted
    df["pred_y"] = df["predicted_label"].astype(str).str.upper().str.strip()
    df = df[df["pred_y"].isin(["REAL", "FAKE"])].copy()

    # Actual (bucket -> label)
    df["actual_bucket"] = df["actual_bucket"].astype(str).str.lower().str.strip()
    df["actual_y"] = df["actual_bucket"].map({"real": "REAL", "misinfo": "FAKE"})

    df = df[df["actual_y"].isin(["REAL", "FAKE"])].copy()
    return df


def confusion_counts(df: pd.DataFrame) -> dict:
    """
    Returns TN, FP, FN, TP using:
      Positive class = FAKE
      Negative class = REAL
    """
    actual = df["actual_y"]
    pred = df["pred_y"]

    tn = int(((actual == "REAL") & (pred == "REAL")).sum())
    fp = int(((actual == "REAL") & (pred == "FAKE")).sum())
    fn = int(((actual == "FAKE") & (pred == "REAL")).sum())
    tp = int(((actual == "FAKE") & (pred == "FAKE")).sum())

    return {"TN": tn, "FP": fp, "FN": fn, "TP": tp}


def confusion_matrix_df(counts: dict) -> pd.DataFrame:
    """
    Pretty 2x2 matrix:
                 Pred REAL   Pred FAKE
      Actual REAL    TN        FP
      Actual FAKE    FN        TP
    """
    return pd.DataFrame(
        [[counts["TN"], counts["FP"]],
         [counts["FN"], counts["TP"]]],
        index=["Actual REAL", "Actual FAKE"],
        columns=["Pred REAL", "Pred FAKE"]
    )


def metrics_from_counts(c: dict) -> dict:
    tn, fp, fn, tp = c["TN"], c["FP"], c["FN"], c["TP"]
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")                  # False Positive Rate on REAL
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")                  # Miss rate on FAKE bucket
    precision = tp / (tp + fp) if (tp + fp) else float("nan")            # Precision for FAKE predictions
    recall = tp / (tp + fn) if (tp + fn) else float("nan")               # Recall for FAKE bucket
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")          # TNR
    balanced_acc = 0.5 * (recall + specificity) if (recall == recall and specificity == specificity) else float("nan")

    return {
        "n": total,
        "accuracy": accuracy,
        "false_positive_rate_on_real": fpr,
        "false_negative_rate_on_fake_bucket": fnr,
        "precision_fake": precision,
        "recall_fake": recall,
        "specificity_real": specificity,
        "balanced_accuracy": balanced_acc,
    }


def print_block(title: str):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"predictions.csv not found at: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Required columns sanity check
    required = {"actual_bucket", "predicted_label", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions.csv missing required columns: {sorted(missing)}")

    df = normalize_labels(df)

    # -----------------------
    # Confusion Matrix #1: Overall (REAL vs MISINFO bucket)
    # -----------------------
    print_block("Confusion Matrix — Overall (REAL vs MISINFO bucket)")
    counts = confusion_counts(df)
    print(confusion_matrix_df(counts).to_string())
    m = metrics_from_counts(counts)
    print("\nMetrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

    # -----------------------
    # Confusion Matrix #2: REAL-only (shows FP/TN behavior)
    # -----------------------
    print_block("Confusion Matrix — REAL-only (focus on false positives)")
    real_df = df[df["actual_y"] == "REAL"].copy()
    real_counts = confusion_counts(real_df)
    print(confusion_matrix_df(real_counts).to_string())
    real_m = metrics_from_counts(real_counts)
    print("\nMetrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in real_m.items()})

    # -----------------------
    # Confusion Matrix #3: Per-source (useful for debugging)
    # -----------------------
    print_block("Confusion Matrices — Per Source")
    for src, g in df.groupby("source"):
        c = confusion_counts(g)
        mm = metrics_from_counts(c)
        print(f"\nSource: {src} (n={mm['n']}, acc={mm['accuracy']:.3f}, FPR_real={mm['false_positive_rate_on_real']:.3f}, recall_fake={mm['recall_fake']:.3f})")
        print(confusion_matrix_df(c).to_string())

    # -----------------------
    # Optional: list your false positives on REAL
    # -----------------------
    print_block("False Positives on REAL (Actual REAL, Pred FAKE)")
    fps = df[(df["actual_y"] == "REAL") & (df["pred_y"] == "FAKE")].copy()
    cols = [c for c in ["source", "title", "url", "full_text_fake_prob", "word_count"] if c in fps.columns]
    if len(fps) == 0:
        print("None 🎉")
    else:
        # Print a compact view
        print(fps[cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
