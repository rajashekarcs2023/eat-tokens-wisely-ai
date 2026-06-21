"""Prove the diversity mechanism: when context contains near-duplicate spans,
the redundancy-penalised selector (beta>0) refuses to spend budget on copies,
while pure relevance greedy (beta=0) piles them up. On non-redundant context the
two coincide (no regression). This is the honest basis for the coverage claim.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from suffix.coverage import select_coverage
from suffix.tokens import count_tokens


def _unit(uid, para, sent, text):
    return {"uid": uid, "para": para, "sent": sent, "n_sent_in_para": 1,
            "title": "t", "text": text, "tokens": count_tokens(text)}


def _redundant_example():
    # Near-duplicate "eiffel" spam scores highest; everest/pacific are distinct.
    # Lengths are comparable so the redundancy penalty (not cost) drives diversity.
    units = [
        _unit("a0", 0, 0, "Quarterly revenue rose twelve percent in the third fiscal quarter."),
        _unit("a1", 1, 0, "Quarterly revenue rose twelve percent in the third fiscal quarter overall."),
        _unit("a2", 2, 0, "Quarterly revenue rose twelve percent in the third fiscal quarter indeed."),
        _unit("b0", 3, 0, "Customer churn fell sharply after the loyalty program launched."),
        _unit("c0", 4, 0, "Warehouse robots doubled outbound shipping throughput last month."),
    ]
    scores = np.array([0.95, 0.93, 0.91, 0.60, 0.55])  # near-dup spam dominates by score
    return {"question": "landmarks", "n_paras": 5, "units": units}, scores


def _n_eiffel(sel):
    return sum(1 for u in sel if u in {"a0", "a1", "a2"})


def test_diversity_skips_duplicates():
    ex, scores = _redundant_example()
    budget = sum(u["tokens"] for u in ex["units"][:3]) + 4  # room for ~3 spans

    sel_greedy = select_coverage(ex, scores, budget, tau=2.0)   # tau>=1 -> plain top-k
    sel_div = select_coverage(ex, scores, budget, tau=0.8)      # near-dup suppression

    assert _n_eiffel(sel_greedy) >= 2, f"pure top-k should hoard duplicates, got {sel_greedy}"
    assert _n_eiffel(sel_div) <= 1, f"suppression must skip duplicates, got {sel_div}"
    assert "b0" in sel_div and "c0" in sel_div, f"must keep distinct facts, got {sel_div}"
    print(f"top-k dups={_n_eiffel(sel_greedy)} {sel_greedy} | dedup dups={_n_eiffel(sel_div)} {sel_div}")


def test_no_regression_without_redundancy():
    units = [
        _unit("u0", 0, 0, "Alpha discovered the polymerase enzyme in 1956."),
        _unit("u1", 1, 0, "Beta charted the Mariana Trench depth precisely."),
        _unit("u2", 2, 0, "Gamma composed the third symphony in Vienna."),
        _unit("u3", 3, 0, "Delta painted the harbor at dawn in oils."),
    ]
    ex = {"question": "facts", "n_paras": 4, "units": units}
    scores = np.array([0.9, 0.7, 0.5, 0.3])
    budget = units[0]["tokens"] + units[1]["tokens"] + 2
    a = select_coverage(ex, scores, budget, tau=2.0)   # plain top-k
    b = select_coverage(ex, scores, budget, tau=0.8)   # with suppression active
    assert set(a) == set(b), f"no redundancy -> identical selection, got {a} vs {b}"
    print(f"non-redundant identical: {sorted(b)}")


if __name__ == "__main__":
    test_diversity_skips_duplicates()
    test_no_regression_without_redundancy()
    print("OK coverage")
