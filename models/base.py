"""
Base class for all VLM inference wrappers.

Every model must implement:
  - load_model()
  - predict(image_path, question) -> str

Everything else (batching, logging, yes/no parsing, prompt formatting)
is handled here so all models behave identically.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import json
import logging
import time
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ── Prompt template ──────────────────────────────────────────────────────────
# Single template used by ALL models. Never change per-model.
# Reviewer-safe: same prompt = no prompt-engineering confound.

POPE_PROMPT = "Is there a {object} in the image? Please answer Yes or No."

ATTRIBUTE_PROMPT = "Is the {object} in the image {attribute}? Please answer Yes or No."

RELATION_PROMPT = (
    "Is the {object_a} {relation} the {object_b} in the image? "
    "Please answer Yes or No."
)


def build_prompt(question_type: str, **kwargs) -> str:
    """
    Build the exact prompt string for a given question type.

    Args:
        question_type: one of 'existence', 'attribute', 'relation'
        **kwargs: fields required by the template

    Returns:
        Formatted prompt string
    """
    if question_type == "existence":
        return POPE_PROMPT.format(**kwargs)
    elif question_type == "attribute":
        return ATTRIBUTE_PROMPT.format(**kwargs)
    elif question_type == "relation":
        return RELATION_PROMPT.format(**kwargs)
    else:
        raise ValueError(f"Unknown question_type: {question_type}")


# ── Answer parser ─────────────────────────────────────────────────────────────

def parse_yes_no(raw: str) -> str:
    """
    Parse a model's raw response into 'yes', 'no', or 'unknown'.

    Strategy (in order):
      1. Strip and lowercase
      2. Check if first word is yes/no
      3. Search for yes/no anywhere in first 10 tokens
      4. Fall back to 'unknown'

    This is intentionally conservative — we never guess.
    Unknown responses are logged and excluded from metrics.
    """
    if not raw or not isinstance(raw, str):
        return "unknown"

    cleaned = raw.strip().lower()

    # Direct match on first word
    first_word = cleaned.split()[0] if cleaned.split() else ""
    first_word = re.sub(r"[^a-z]", "", first_word)
    if first_word == "yes":
        return "yes"
    if first_word == "no":
        return "no"

    # Search in first 10 tokens
    tokens = cleaned.split()[:10]
    for tok in tokens:
        tok_clean = re.sub(r"[^a-z]", "", tok)
        if tok_clean == "yes":
            return "yes"
        if tok_clean == "no":
            return "no"

    return "unknown"


# ── Base class ────────────────────────────────────────────────────────────────

class BaseVLM(ABC):
    """
    Abstract base for all VLM wrappers in this project.

    Subclasses must implement:
      - load_model(): load weights, processor, tokenizer into self
      - predict(image_path, question): return raw string response

    Subclasses should NOT override:
      - predict_batch()
      - evaluate_file()
      - Any prompt/parsing logic
    """

    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """
        Args:
            model_name: short identifier used in output filenames
                        e.g. 'qwen2vl_7b', 'internvl2_8b'
            device:     'cuda' or 'cpu'
            **kwargs:   passed through to load_model()
        """
        self.model_name = model_name
        self.device = device
        self.logger = logging.getLogger(model_name)
        self._loaded = False
        self._load_kwargs = kwargs

    # ── Must implement ────────────────────────────────────────────────────────

    @abstractmethod
    def load_model(self, **kwargs):
        """
        Load model weights, processor, tokenizer into self.
        Called lazily on first use by ensure_loaded().

        Store everything as instance attributes, e.g.:
          self.model = ...
          self.processor = ...
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: str | Path, question: str) -> str:
        """
        Run a single inference.

        Args:
            image_path: path to image file (JPEG/PNG)
            question:   exact prompt string (from build_prompt())

        Returns:
            Raw string response from the model.
            Do NOT parse yes/no here — return the model's raw output.
        """
        raise NotImplementedError

    # ── Provided by base ──────────────────────────────────────────────────────

    def ensure_loaded(self):
        """Lazy loader — call this at the start of predict_batch()."""
        if not self._loaded:
            self.logger.info("Loading model...")
            t0 = time.time()
            self.load_model(**self._load_kwargs)
            self._loaded = True
            self.logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    def predict_batch(
        self,
        items: list[dict],
        image_dir: str | Path,
    ) -> list[dict]:
        """
        Run inference on a list of benchmark items.

        Args:
            items: list of dicts, each must have:
                   - 'image': filename (relative to image_dir)
                   - 'question': full prompt string
                   - 'label': ground truth ('yes' or 'no')
                   - 'question_id': unique identifier
                   Optional:
                   - 'question_type': 'existence'|'attribute'|'relation'
                   - any other metadata fields (preserved in output)

            image_dir: root directory containing image files

        Returns:
            Same list with added fields per item:
                - 'raw_answer': model's raw string
                - 'answer': parsed 'yes'/'no'/'unknown'
                - 'correct': bool (None if answer == 'unknown')
                - 'latency_ms': inference time in milliseconds
        """
        self.ensure_loaded()
        image_dir = Path(image_dir)
        results = []

        for i, item in enumerate(items):
            image_path = image_dir / item["image"]

            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}")
                item = {**item, "raw_answer": "", "answer": "unknown",
                        "correct": None, "latency_ms": 0}
                results.append(item)
                continue

            t0 = time.time()
            try:
                raw = self.predict(image_path, item["question"])
            except Exception as e:
                self.logger.error(f"Inference failed on {item['question_id']}: {e}")
                raw = ""

            latency_ms = (time.time() - t0) * 1000
            answer = parse_yes_no(raw)
            label = item["label"].strip().lower()

            correct = None
            if answer != "unknown":
                correct = answer == label

            result = {
                **item,
                "raw_answer": raw,
                "answer": answer,
                "correct": correct,
                "latency_ms": round(latency_ms, 1),
            }
            results.append(result)

            if (i + 1) % 100 == 0:
                n_done = i + 1
                n_correct = sum(r["correct"] for r in results if r["correct"] is not None)
                n_valid = sum(1 for r in results if r["correct"] is not None)
                acc = n_correct / n_valid if n_valid > 0 else 0
                self.logger.info(
                    f"[{n_done}/{len(items)}] running acc={acc:.3f} "
                    f"unknown={n_done - n_valid}"
                )

        return results

    def evaluate_file(
        self,
        input_path: str | Path,
        image_dir: str | Path,
        output_path: str | Path,
    ) -> dict:
        """
        Full evaluation pipeline: load JSONL → infer → save results.

        Args:
            input_path:  path to benchmark JSONL file
            image_dir:   directory containing images
            output_path: where to write predictions JSONL

        Returns:
            Summary dict with keys: model, benchmark, n_total, n_valid,
            n_unknown, accuracy, yes_rate, latency_mean_ms
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load benchmark
        items = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        self.logger.info(
            f"Evaluating {self.model_name} on {input_path.name} "
            f"({len(items)} items)"
        )

        # Run inference
        results = self.predict_batch(items, image_dir)

        # Save predictions
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Compute summary
        n_total = len(results)
        valid = [r for r in results if r["correct"] is not None]
        n_valid = len(valid)
        n_unknown = n_total - n_valid

        accuracy = sum(r["correct"] for r in valid) / n_valid if n_valid > 0 else 0
        yes_rate = (
            sum(1 for r in valid if r["answer"] == "yes") / n_valid
            if n_valid > 0 else 0
        )
        latency_mean = sum(r["latency_ms"] for r in results) / n_total

        summary = {
            "model": self.model_name,
            "benchmark": input_path.stem,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_unknown": n_unknown,
            "accuracy": round(accuracy, 4),
            "yes_rate": round(yes_rate, 4),
            "latency_mean_ms": round(latency_mean, 1),
        }

        self.logger.info(
            f"Done — acc={accuracy:.3f} yes_rate={yes_rate:.3f} "
            f"unknown={n_unknown}/{n_total}"
        )

        return summary

    def __repr__(self):
        status = "loaded" if self._loaded else "not loaded"
        return f"{self.__class__.__name__}(name={self.model_name}, {status})"
