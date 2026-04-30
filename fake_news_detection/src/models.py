from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Curated Hugging Face models — pick based on language + use-case.
# ---------------------------------------------------------------------------
AVAILABLE_HF_MODELS = {
    # Indonesian
    "indobert": "indolem/indobert-base-uncased",
    "indobert-sentiment": "mdhugol/indonesia-bert-sentiment-classification",
    "xlm-roberta-id": "cahya/xlm-roberta-base-indonesian-NER",  # embedding fallback
    # English fake-news specific
    "roberta-fake-news": "hamzab/roberta-fake-news-classification",
    "bert-fake-news": "jy46604790/Fake-News-Bert-Detect",
    # Multilingual
    "xlm-roberta": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
}


# ============================================================================
# Classical ML
# ============================================================================

@dataclass
class ClassicalMLModel:
    """TF-IDF + sklearn classifier (Logistic Regression / SVM / RF / XGB).

    Example
    -------
    >>> m = ClassicalMLModel(kind="logreg")
    >>> m.fit(X_train, y_train)
    >>> m.predict(X_test)
    """
    kind: str = "logreg"  # "logreg" | "svm" | "rf" | "xgb"
    max_features: int = 20000
    ngram_range: tuple = (1, 2)
    pipeline: object = field(init=False, default=None)
    classes_: Optional[np.ndarray] = field(init=False, default=None)

    def _build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        vec = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=True,
        )

        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        elif self.kind == "svm":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV
            clf = CalibratedClassifierCV(LinearSVC(), cv=3)
        elif self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        elif self.kind == "xgb":
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1,
                n_jobs=-1, random_state=42, eval_metric="logloss",
            )
        else:
            raise ValueError(f"Unknown kind {self.kind!r}")

        self.pipeline = Pipeline([("tfidf", vec), ("clf", clf)])

    def fit(self, X: Sequence[str], y: Sequence) -> "ClassicalMLModel":
        if self.pipeline is None:
            self._build()
        self.pipeline.fit(list(X), list(y))
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        return self

    def predict(self, X: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict(list(X))

    def predict_proba(self, X: Sequence[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(X))


# ============================================================================
# Hugging Face
# ============================================================================

class HuggingFaceClassifier:
    """Thin wrapper around `transformers.pipeline('text-classification')`.

    - Supports pre-trained models (zero-shot prediction via `.predict`).
    - Supports fine-tuning via `.fine_tune` using `Trainer`.
    """

    def __init__(
        self,
        model_id: str = AVAILABLE_HF_MODELS["roberta-fake-news"],
        device: Optional[int] = None,
        max_length: int = 256,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.device = device
        self._pipe = None
        self._label_map: Optional[dict] = None

    # -- inference ---------------------------------------------------------

    def _ensure(self):
        if self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "text-classification",
            model=self.model_id,
            device=self.device if self.device is not None else -1,
            truncation=True,
            max_length=self.max_length,
            top_k=None,
        )

    def predict(self, X: Iterable[str]) -> List[str]:
        self._ensure()
        preds = []
        for t in X:
            out = self._pipe(t)
            if isinstance(out[0], list):
                out = out[0]
            out = sorted(out, key=lambda d: d["score"], reverse=True)
            preds.append(out[0]["label"])
        return preds

    def predict_proba(self, X: Iterable[str]) -> np.ndarray:
        """Return a 2D array [n_samples, n_labels] in the pipeline's label order."""
        self._ensure()
        rows: List[List[float]] = []
        label_order: Optional[List[str]] = None
        for t in X:
            out = self._pipe(t)
            if isinstance(out[0], list):
                out = out[0]
            scores = {d["label"]: float(d["score"]) for d in out}
            if label_order is None:
                label_order = list(scores.keys())
            rows.append([scores.get(l, 0.0) for l in label_order])
        self._label_map = {i: l for i, l in enumerate(label_order or [])}
        return np.array(rows)

    # -- fine-tuning -------------------------------------------------------

    def fine_tune(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        val_texts: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[int]] = None,
        output_dir: str = "./hf_finetuned",
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        num_labels: int = 2,
    ):
        """Fine-tune the base model on a labelled dataset.

        This downloads the raw model (not a pipeline-wrapped one) and
        uses HF Trainer. Requires `datasets` and `accelerate`.
        """
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
        )
        import torch
        from torch.utils.data import Dataset

        tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, num_labels=num_labels
        )

        class _DS(Dataset):
            def __init__(self, texts, labels):
                self.enc = tok(
                    list(texts), truncation=True, padding=False,
                    max_length=256,
                )
                self.labels = list(labels)

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, i):
                item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
                item["labels"] = torch.tensor(int(self.labels[i]))
                return item

        train_ds = _DS(train_texts, train_labels)
        val_ds = _DS(val_texts, val_labels) if val_texts is not None else None

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            evaluation_strategy="epoch" if val_ds else "no",
            save_strategy="epoch",
            logging_steps=50,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tok,
            data_collator=DataCollatorWithPadding(tok),
        )
        trainer.train()
        trainer.save_model(output_dir)
        tok.save_pretrained(output_dir)
        # swap in the fine-tuned model for subsequent .predict calls
        self.model_id = output_dir
        self._pipe = None
        return trainer
