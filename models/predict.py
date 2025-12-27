from cProfile import label
import os
import json
from pydoc import text
import re
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------
# Config
# -----------------------
# UPDATED: Use absolute path or relative to project root if possible, 
# but for now we keep the default. 
# In a real app, strict paths or env vars are better.
MODEL_DIR = "models/roberta_text_v1"
MAX_LEN = 512  # keep inference snappy; you can raise later
DEFAULT_TOP_K = 6
FAKE_THRESHOLD = 0.5  # aggressive mode


class FakeNewsPredictor:
    """
    Encapsulates the model, tokenizer, and prediction logic.
    """
    def __init__(self, model_dir: str = MODEL_DIR, device: str = None):
        self.model_dir = model_dir
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Loading model from {self.model_dir} to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()
        self.temperature = self._load_temperature()
        print(f"Model loaded. Temperature: {self.temperature}")

    def _load_temperature(self) -> float:
        temp_path = os.path.join(self.model_dir, "temperature.json")
        if os.path.exists(temp_path):
            with open(temp_path, "r") as f:
                return float(json.load(f).get("temperature", 1.0))
        return 1.0

    def _softmax_fake_prob(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [B,2]
        probs = torch.softmax(logits / self.temperature, dim=1)
        return probs[:, 1]  # fake prob

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Lightweight sentence splitter. Good enough for app UX.
        """
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        
        # split on sentence end punctuation with trailing space
        parts = re.split(r"(?<=[.!?])\s+", text)
        # fall back if no punctuation
        if len(parts) == 1:
            # chunk by length
            return self._chunk_by_words(text, max_words=60)
        return [p.strip() for p in parts if p.strip()]

    def _chunk_by_words(self, text: str, max_words: int = 60) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _batch_predict_probs(self, texts: List[str]) -> np.ndarray:
        """
        Returns p_fake for each text snippet.
        """
        if len(texts) == 0:
            return np.array([], dtype=float)

        enc = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self.model(**enc).logits
            p_fake = self._softmax_fake_prob(logits).detach().cpu().numpy()
        return p_fake

    def _highlight_text_snippets(self, full_text: str, snippets: List[str]) -> str:
        """
        Highlight snippets by surrounding them with [[...]].
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

    def predict(self, text: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Refactored main prediction logic returning a dictionary.
        """
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("Empty input text.")

        # Overall score (single pass on full text)
        # Note: In original code, it calls batch_predict_probs which calls tokenizer which adds special tokens.
        # The input text to batch_predict_probs in original code was just the raw text (cleaned).
        # Wait, original code in split_sentences added "TITLE: BODY:". 
        # But for OVERALL score, the original code did:
        # overall_p = float(batch_predict_probs(..., [text], ...)[0])
        # And `text` there was just `user_text` (cleaned). 
        # It did NOT add "TITLE: BODY:" for the overall score call?
        # Checking lines 42-47 of original:
        # split_into_sentences ADDS "TITLE: BODY:".
        # predict_text lines 137-142: cleans text, then calls batch_predict_probs([text]). 
        # So overall score DOES NOT have "TITLE: BODY:" prepended?
        # That seems inconsistent if the training data had it.
        # Let's check `clean_df.py`:
        # df["input_text"] = "TITLE: " + df["title"] + " BODY: " + df["text"]
        
        # So the model was trained with "TITLE: ... BODY: ...".
        # If the original `predict.py` wasn't adding it for the overall score, that might be a bug in the old script 
        # or `text` input was expected to include it?
        # The original `predict.py` line 47: `text = "TITLE:  BODY: " + text` is inside `split_into_sentences`.
        # But `predict_text` calls `batch_predict_probs` on the raw `text` first (line 142).
        
        # I will FIX this by ensuring "TITLE: BODY: " is prepended if not present, to match training.
        # But to be safe and avoid breaking changes if `text` already has it, I'll simple prepend it for now
        # OR assume the user input is just the body. 
        # I'll stick to what seemed to be the intent: PREPEND it for consistency.
        
        input_text_formatted = f"TITLE:  BODY: {text}"
        
        overall_p = float(self._batch_predict_probs([input_text_formatted])[0])
        
        # Evidence: sentence/chunk scoring
        # split_into_sentences adds the prefix internally in the original code. 
        # My refactored _split_into_sentences also adds it.
        snippets = self._split_into_sentences(text)
        snippets = [s for s in snippets if len(s.split()) >= 5]
        
        # Calculate snippet probs - Prefix for the model but keep original for highlighting
        snippets_prefixed = [f"TITLE:  BODY: {s}" for s in snippets]
        snippet_probs = self._batch_predict_probs(snippets_prefixed)
        
        pairs = list(zip(snippets, snippet_probs.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        
        top_pairs = pairs[:top_k]
        top_pairs = [(s, p) for (s, p) in top_pairs if len(s.split()) >= 5]
        top_snips = [s for (s, _) in top_pairs]
        
        highlighted = self._highlight_text_snippets(text, top_snips)
        
        top_probs = [p for _, p in top_pairs]
        avg_top3 = float(np.mean(top_probs[:3])) if len(top_probs) >= 3 else (float(np.mean(top_probs)) if top_probs else 0.0)
        
        final_p = max(overall_p, 0.6 * avg_top3)
        label = "FAKE" if final_p >= 0.65 else "REAL" # Threshold from original code line 188
        
        return {
            "label": label,
            "probability": final_p,
            "overall_prediction": overall_p,
            "highlighted_text": highlighted,
            "evidence": [{"text": s, "score": p} for s, p in top_pairs]
        }


# Global instance for CLI reuse
# We don't initialize it at module level to avoid loading on import
# unless we want to.
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="How many snippets to highlight")
    args = parser.parse_args()

    # Load predictor
    predictor = FakeNewsPredictor()

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

    result = predictor.predict(user_text, top_k=args.top_k)

    print("\n====================")
    print("Prediction")
    print("====================")
    print(f"Label: {result['label']}")
    print(f"Fake probability: {result['probability']*100:.2f}%")
    print(f"(debug) full-text fake probability: {result['overall_prediction']*100:.2f}%")

    print("\n====================")
    print("Highlighted (most 'fake-leaning' snippets)")
    print("====================")
    print(result['highlighted_text'])

    print("\n====================")
    print("Why (evidence snippets + scores)")
    print("====================")
    for i, item in enumerate(result['evidence'], start=1):
        print(f"{i}. ({item['score']*100:.1f}% fake) {item['text']}")

    print("\nNote: These highlights are not a factual proof—just the text regions that most increased the model's fake score.")


if __name__ == "__main__":
    main()

