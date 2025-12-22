from cProfile import label
import os
import json
from pydoc import text
import re
import argparse
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------
# Config
# -----------------------
MODEL_DIR = "models/roberta_text_v1"
MAX_LEN = 512  # keep inference snappy; you can raise later
DEFAULT_TOP_K = 6


FAKE_THRESHOLD = 0.5  # aggressive mode

def decide_label(p_fake: float) -> str:
    return "FAKE" if p_fake >= FAKE_THRESHOLD else "REAL"


def load_temperature(model_dir: str) -> float:
    temp_path = os.path.join(model_dir, "temperature.json")
    if os.path.exists(temp_path):
        with open(temp_path, "r") as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0


def softmax_fake_prob(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    # logits: [B,2]
    probs = torch.softmax(logits / temperature, dim=1)
    return probs[:, 1]  # fake prob


def split_into_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter. Good enough for app UX.
    """
    text = re.sub(r"\s+", " ", text).strip()
    text = "TITLE:  BODY: " + text
    if not text:
        return []
    # split on sentence end punctuation with trailing space
    parts = re.split(r"(?<=[.!?])\s+", text)
    # fall back if no punctuation
    if len(parts) == 1:
        # chunk by length
        return chunk_by_words(text, max_words=60)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_words(text: str, max_words: int = 60) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def batch_predict_probs(
    tokenizer,
    model,
    temperature: float,
    texts: List[str],
    device: torch.device,
) -> np.ndarray:
    """
    Returns p_fake for each text snippet.
    """
    if len(texts) == 0:
        return np.array([], dtype=float)

    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        p_fake = softmax_fake_prob(logits, temperature).detach().cpu().numpy()
    return p_fake


def highlight_text_snippets(full_text: str, snippets: List[str]) -> str:
    """
    Highlight snippets by surrounding them with [[...]].
    Uses a simple exact-ish match (case-insensitive) for readability.
    """
    highlighted = full_text
    for snip in snippets:
        if len(snip) < 8:
            continue
        # escape for regex, case-insensitive replacement once
        pattern = re.escape(snip)
        # try exact match first; if not found, try a loosened version
        m = re.search(pattern, highlighted, flags=re.IGNORECASE)
        if m:
            start, end = m.span()
            highlighted = highlighted[:start] + "[[" + highlighted[start:end] + "]]" + highlighted[end:]
        else:
            # loose match: remove non-word chars and try again
            loose = re.sub(r"\W+", r"\\W+", re.escape(snip))
            m2 = re.search(loose, highlighted, flags=re.IGNORECASE)
            if m2:
                start, end = m2.span()
                highlighted = highlighted[:start] + "[[" + highlighted[start:end] + "]]" + highlighted[end:]
    return highlighted


def predict_text(
    tokenizer,
    model,
    temperature: float,
    text: str,
    device: torch.device,
    top_k: int = DEFAULT_TOP_K
) -> Tuple[float, float, List[Tuple[str, float]], str, str]:
    """
    Returns:
    - overall p_fake (0..1)
    - list of (snippet, snippet_p_fake) sorted by most fake
    - highlighted text
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Empty input text.")

    # Overall score (single pass on full text)
    overall_p = float(batch_predict_probs(tokenizer, model, temperature, [text], device)[0])
    label = decide_label(overall_p)

    # DEBUG: how many non-padding tokens the model is actually seeing
    enc_full = tokenizer(
    [text],
    truncation=True,
    padding="max_length",
    max_length=MAX_LEN,
    return_tensors="pt"
    )
    full_nonpad = int((enc_full["attention_mask"][0] == 1).sum().item())
    print(f"[DEBUG] full_text nonpad tokens: {full_nonpad} / {MAX_LEN}")


    # Evidence: sentence/chunk scoring
    snippets = split_into_sentences(text)
    snippets = [s for s in snippets if len(s.split()) >= 5]
    # If very long, reduce number of snippets by chunking
    if snippets:
        enc_s0 = tokenizer(
        [snippets[0]],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    s0_nonpad = int((enc_s0["attention_mask"][0] == 1).sum().item())
    print(f"[DEBUG] snippet0 nonpad tokens: {s0_nonpad} / {MAX_LEN}")

    snippet_probs = batch_predict_probs(tokenizer, model, temperature, snippets, device)

    pairs = list(zip(snippets, snippet_probs.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)

    top_pairs = pairs[:top_k]
    top_pairs = [(s, p) for (s, p) in top_pairs if len(s.split()) >= 5]
    top_snips = [s for (s, _) in top_pairs]

    highlighted = highlight_text_snippets(text, top_snips)


    top_probs = [p for _, p in top_pairs]
    avg_top3 = float(np.mean(top_probs[:3])) if len(top_probs) >= 3 else (float(np.mean(top_probs)) if top_probs else 0.0)

    final_p = max(overall_p, 0.6 * avg_top3)
    label = "FAKE" if final_p >= 0.65 else "REAL"
    return overall_p, final_p, top_pairs, highlighted, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="How many snippets to highlight")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    temperature = load_temperature(MODEL_DIR)
    print(f"Loaded model: {MODEL_DIR}")
    print(f"Loaded temperature: {temperature}\n")

    print("Paste text, then press Enter. End input with Ctrl-D (mac) or Ctrl-Z then Enter (windows).\n")
    try:
        user_text = ""
        while True:
            line = input()
            user_text += line + "\n"
    except EOFError:
        pass

    user_text = user_text.strip()
    if not user_text:
        print("No text received.")
        return

    overall_p, final_p, top_pairs, highlighted, label = predict_text(
        tokenizer=tokenizer,
        model=model,
        temperature=temperature,
        text=user_text,
        device=device,
        top_k=args.top_k
    )

    print("\n====================")
    print("Prediction")
    print("====================")
    print(f"Label: {label}")
    print(f"Fake probability: {final_p*100:.2f}%")           # main number shown to user
    print(f"(debug) full-text fake probability: {overall_p*100:.2f}%")

    print("\n====================")
    print("Highlighted (most 'fake-leaning' snippets)")
    print("====================")
    print(highlighted)

    print("\n====================")
    print("Why (evidence snippets + scores)")
    print("====================")
    for i, (snip, p) in enumerate(top_pairs, start=1):
        print(f"{i}. ({p*100:.1f}% fake) {snip}")

    print("\nNote: These highlights are not a factual proof—just the text regions that most increased the model's fake score.")


if __name__ == "__main__":
    main()
