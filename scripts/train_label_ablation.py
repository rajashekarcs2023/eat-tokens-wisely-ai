"""Train the three label-source scorers (SPEC v2 §5): A / B / C.

Everything is held identical — same train examples, same features
(suffix.features.example_features), same LogReg pipeline — and ONLY the training
LABEL differs:
  A = is_gold (human supporting_facts)
  B = importance (LLMLingua-2-style, query-agnostic, no reader)
  C = reader-grounded answer-impact (leave-one-out minimal sufficient gold)

To be fair, all three train on the SAME example set (the examples where C is
valid). Saves scorer_{A,B,C}.joblib loadable by suffix.scorer.KeepScorer.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.features import FEATURE_NAMES, example_features  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def make_pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ])


def label_for(ex, source, imp, res):
    units = ex["units"]
    if source == "A":
        return np.array([int(u["is_gold"]) for u in units])
    if source == "B":
        d = imp.get(ex["id"], {}).get("labels", {})
        return np.array([int(d.get(u["uid"], 0)) for u in units])
    if source == "C":
        d = res.get(ex["id"], {}).get("labels", {})
        return np.array([int(d.get(u["uid"], 0)) for u in units])
    raise ValueError(source)


def main():
    train = json.load(open(os.path.join(DATA, "hotpot_train.json")))
    test = json.load(open(os.path.join(DATA, "hotpot_test.json")))
    imp = json.load(open(os.path.join(DATA, "importance_labels_train.json")))
    res = json.load(open(os.path.join(DATA, "residual_labels_train.json")))

    # fair: train on the examples where C is valid (has labels)
    valid_ids = {eid for eid, r in res.items() if "labels" in r and not r.get("skip")}
    train_ex = [ex for ex in train if ex["id"] in valid_ids]
    print(f"training all three scorers on {len(train_ex)} shared examples")

    report = {}
    for source in ["A", "B", "C"]:
        X = np.vstack([example_features(ex) for ex in train_ex])
        y = np.concatenate([label_for(ex, source, imp, res) for ex in train_ex])
        keep_rate = y.mean()
        model = make_pipeline().fit(X, y)
        import joblib
        joblib.dump(model, os.path.join(DATA, f"scorer_{source}.joblib"))

        # diagnostic AUC vs is_gold on TEST (favors A by construction — noted)
        Xt = np.vstack([example_features(ex) for ex in test])
        yt = np.concatenate([[int(u["is_gold"]) for u in ex["units"]] for ex in test])
        proba = model.predict_proba(Xt)[:, 1]
        auc = roc_auc_score(yt, proba)
        report[source] = {"train_keep_rate": float(keep_rate), "auc_vs_isgold_test": float(auc)}
        top = sorted(zip(FEATURE_NAMES, model.named_steps["clf"].coef_[0]), key=lambda t: -abs(t[1]))[:5]
        print(f"  scorer {source}: train_keep_rate={keep_rate:.2f}  AUC-vs-isGold(test)={auc:.3f}  "
              f"top:{[(n, round(w,2)) for n,w in top]}")

    json.dump(report, open(os.path.join(DATA, "ablation_scorers_report.json"), "w"), indent=2)
    print("\nNOTE: AUC-vs-isGold favors A by construction; the REAL comparison is the downstream eval.")
    print("saved scorer_{A,B,C}.joblib + ablation_scorers_report.json")


if __name__ == "__main__":
    main()
