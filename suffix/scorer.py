"""The learned component: a per-span keep-scorer.

A scikit-learn LogisticRegression (StandardScaler + L2) fit on HotpotQA's human
`supporting_facts` labels. This is a compressor *learned from measured ground
truth* — not a hand-weighted formula and not an LLM's opinion. At inference it
emits P(span is decision-relevant | question), in milliseconds, on CPU.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, example_features


class KeepScorer:
    def __init__(self, C: float = 1.0):
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
            ]
        )
        self.fitted = False

    def fit(self, examples):
        X = np.vstack([example_features(e) for e in examples])
        y = np.concatenate([[u["is_gold"] for u in e["units"]] for e in examples]).astype(int)
        self.model.fit(X, y)
        self.fitted = True
        return self

    def score_example(self, example) -> np.ndarray:
        """P(keep) per unit, in example order."""
        feats = example_features(example)
        return self.model.predict_proba(feats)[:, 1]

    def coef_table(self):
        clf = self.model.named_steps["clf"]
        return sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda t: -abs(t[1]))

    def save(self, path):
        import joblib

        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        import joblib

        obj = cls()
        obj.model = joblib.load(path)
        obj.fitted = True
        return obj
