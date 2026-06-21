"""Variable-rate GATE (no API): is there headroom for adaptive per-example budget
allocation over a fixed budget?

From the per-example rate-distortion curves (ratesweep.json), compute:
  - FIXED frontier: every example gets the same budget.
  - ORACLE adaptive frontier: each example gets its own budget, optimally allocated
    under an average-token constraint (Lagrangian water-filling over the RD curves).
Gate: if oracle clearly beats fixed at matched average tokens, build the learned
predictor. Otherwise the idea has no headroom -- stop and report honestly.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    sweep = json.load(open(os.path.join(DATA, "ratesweep.json")))
    per = sweep["per_example"]
    budgets = [str(b) for b in sweep["budgets"]] + ["full"]
    keys = list(per.keys())

    # matrices: f1[ex, b], tok[ex, b]
    F = np.array([[per[k]["f1"].get(b, 0.0) for b in budgets] for k in keys])
    T = np.array([[per[k]["tokens"].get(b, 0.0) for b in budgets] for k in keys])
    n = len(keys)

    # FIXED frontier (one column each)
    fixed = [(T[:, j].mean(), F[:, j].mean(), budgets[j]) for j in range(len(budgets))]

    # ORACLE frontier via Lagrangian: for lambda, each ex picks argmax_b F - lambda*T
    oracle = []
    for lam in np.concatenate([[0.0], np.geomspace(1e-5, 0.05, 60)]):
        choice = np.argmax(F - lam * T, axis=1)
        oracle.append((float(T[np.arange(n), choice].mean()), float(F[np.arange(n), choice].mean())))
    oracle.sort(key=lambda o: o[0])

    def oracle_f1_at(tok_target):
        xs = [o[0] for o in oracle]; ys = [o[1] for o in oracle]
        return float(np.interp(tok_target, xs, ys))

    def oracle_tokens_at(f1_target):
        # min avg tokens on the oracle frontier achieving >= f1_target
        cand = [o[0] for o in oracle if o[1] >= f1_target]
        return float(min(cand)) if cand else None

    print("=== FIXED-budget frontier ===")
    for t, f, b in fixed:
        print(f"  budget {b:>5}: avg_tokens={t:6.0f}  avg_F1={f:.3f}")

    print("\n=== ORACLE adaptive vs FIXED (headroom) ===")
    gains = []
    for t, f, b in fixed:
        if b == "full":
            continue
        of = oracle_f1_at(t)
        print(f"  @ ~{t:4.0f} avg tok:  fixed_F1={f:.3f}   oracle_F1={of:.3f}   headroom=+{of-f:.3f}")
        gains.append(of - f)
    # token savings at matched quality (use fixed-240 F1 as the operating point)
    f240 = next(f for t, f, b in fixed if b == "240")
    t240 = next(t for t, f, b in fixed if b == "240")
    ot = oracle_tokens_at(f240)
    print(f"\n  to MATCH fixed@240 F1={f240:.3f} ({t240:.0f} tok): oracle needs ~{ot:.0f} avg tok "
          f"({(1-ot/t240)*100:.0f}% fewer)" if ot else "  (n/a)")

    max_gain = max(gains) if gains else 0.0
    print(f"\n  >>> max headroom = +{max_gain:.3f} F1 at matched tokens")
    gate = max_gain >= 0.02 or (ot and ot <= 0.85 * t240)
    print("  GATE:", "PASS ✅ — build the learned budget predictor." if gate else
          "FAIL ❌ — no meaningful headroom; stop and report honestly.")

    json.dump({"fixed": [(float(t), float(f), b) for t, f, b in fixed],
               "oracle": [(float(t), float(f)) for t, f in oracle],
               "max_headroom": float(max_gain), "gate_pass": bool(gate),
               "match240_oracle_tokens": ot, "fixed240_tokens": float(t240), "fixed240_f1": float(f240)},
              open(os.path.join(DATA, "adaptive_gate.json"), "w"))
    print("\nsaved data/adaptive_gate.json")


if __name__ == "__main__":
    main()
