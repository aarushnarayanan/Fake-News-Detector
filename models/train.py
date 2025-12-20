#total model time for v1 on 1 epoch~54 minutes and 14 seconds~99.97% eval accuracy

import os
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

MODEL_NAME = "roberta-base"
DATA_DIR = "data/tokenized_datasets"
OUT_DIR = "models/roberta_text"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits[:, 1])) if logits.shape[1] == 2 else None
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)

    metrics = {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # Optional but nice: ROC-AUC (only for binary)
    if probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
        except Exception:
            pass

    return metrics

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load tokenized datasets
    ds = load_from_disk(DATA_DIR)
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # RoBERTa doesn't use token_type_ids, so collator is straightforward
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",  # keeps it simple; later you can use "wandb"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\nFinal evaluation on TEST:")
    test_metrics = trainer.evaluate(test_ds)
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # Save model + tokenizer for inference in your backend
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved model + tokenizer to: {OUT_DIR}")

if __name__ == "__main__":
    main()
