"""Phase 0 check 0.5 (Experiment 6) and the Astra reference numbers on our opponents.

  .venv/bin/python -m wsp.go.phase0            -> results/phase0/go_check.json

1. Head-to-head, 2,000 games (seed 31337, the written network Black in the first half): Astra's budget_0 and
   budget_50 networks vs the released same-size PPO networks (lr 3e-5, seeds 0/1/2). Original: 42.6 / 26.3 / 36.3 %
   (budget 0), 74.2 / 70.0 / 78.5 % (budget 50); eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json.
2. Our opponents against a uniform random player (2,000 games), against each other, and the calibration targets
   from the prompt (practice weak / strong 95 / 98 %, hidden 79-98 %).
3. evaluate() of Astra's four networks against OUR hidden opponents (the Astra reference numbers in our frame), next
   to the original eval_submission.json numbers; and of the three released same-size PPO networks (original:
   eval/ppo_same_size/ppo_big_eval/eval_lr_scan.json).
"""
import json
import os
import time

import numpy as np

from . import tools as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGS = os.path.join(ROOT, "lm_understand-logs", "exp6_go_from_scratch")
NETS = os.path.join(LOGS, "networks")
OUT = os.path.join(ROOT, "results", "phase0", "go_check.json")


def se(p, n):
    return float(np.sqrt(p * (1 - p) / n))


def main():
    t0 = time.time()
    out = {"protocol": {"pgx": "2.6.0 go_9x9 (komi 7.5, positional superko over the whole game -> mover loses, "
                               "two passes or 162 plies, area scoring)",
                        "sampling": "temperature 1 over legal moves, no search", "games": 2000,
                        "h2h_seed": 31337, "eval_seeds": "777 + j over (ppo_s0, ppo_s2, reinforce_s0, reinforce_s2, "
                                                         "random)", "backend": "JAX CPU (local)"}}
    # 1. head-to-head vs released same-size PPO
    orig = json.load(open(os.path.join(LOGS, "eval/ppo_same_size/ppo_big_eval/h2h_lr_scan.json")))
    h = {}
    for b, run in (("budget_0", "v1_b0"), ("budget_50", "v1_b50")):
        for s in range(3):
            r = T.h2h(os.path.join(NETS, b), os.path.join(NETS, f"ppo_same_size_lr3e-5_s{s}"), 2000, 31337)
            o = orig[f"{run}_vs_lr3e-5" + ("" if s == 0 else f"_s{s}")]
            z = (r["score"] - o["score"]) / np.sqrt(2) / se(o["score"], 2000)
            h[f"{b}_vs_ppo_same_size_s{s}"] = {"ours": r, "original": o, "diff_pts": round(100 * (r["score"] - o["score"]), 2),
                                              "z": round(float(z), 2)}
            print(b, s, round(r["score"], 4), "orig", round(o["score"], 4), flush=True)
    out["h2h_same_size_ppo"] = h
    # 2. opponents
    opp = {}
    for name in ("ppo_s0", "ppo_s1", "ppo_s2", "reinforce_s0", "reinforce_s1", "reinforce_s2"):
        r = T.h2h(name, "random", 2000, 555)
        info = json.load(open(os.path.join(T.OPP_DIR, name, "info.json")))
        opp[name] = {"role": info["role"], "iteration": info["iteration"], "vs_random": r["score"],
                     "vs_random_summary": r}
        print(name, "vs random", r["score"], flush=True)
    opp["_targets"] = {"practice weak (reinforce_s1)": 0.95, "practice strong (ppo_s1)": 0.98,
                       "hidden": "0.79-0.98 (four networks)"}
    names = ["ppo_s0", "ppo_s1", "ppo_s2", "reinforce_s0", "reinforce_s1", "reinforce_s2"]
    cross = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cross[f"{a}_vs_{b}"] = T.h2h(a, b, 2000, 556)["score"]
    opp["_cross_play"] = cross
    out["opponents"] = opp
    # 3. Astra's networks and the released same-size PPO through the hidden evaluation
    ev = {}
    for b in ("budget_0", "budget_10", "budget_25", "budget_50"):
        r = T.evaluate(os.path.join(NETS, b))
        o = json.load(open(os.path.join(LOGS, "eval", b, "eval_submission.json")))
        ev[b] = {"ours": r, "original_scores": {k: o[k]["score"] for k in T.HIDDEN}, "original_main": o["main_score"]}
        ev[b]["analysis_strong_ours"] = T.analysis_strong(os.path.join(NETS, b))
        print(b, {k: round(r[k]["score"], 3) for k in T.HIDDEN}, "orig",
              {k: round(o[k]["score"], 3) for k in T.HIDDEN}, flush=True)
    lr = json.load(open(os.path.join(LOGS, "eval/ppo_same_size/ppo_big_eval/eval_lr_scan.json")))
    for s in range(3):
        r = T.evaluate(os.path.join(NETS, f"ppo_same_size_lr3e-5_s{s}"))
        ev[f"ppo_same_size_lr3e-5_s{s}"] = {"ours": r, "original_scores": lr[f"lr3e-5_s{s}_iter20000"]}
        print("same-size", s, {k: round(r[k]["score"], 3) for k in T.HIDDEN}, flush=True)
    out["evaluate"] = ev
    out["astra_reference_on_our_opponents"] = {
        b: {"ppo_s0": ev[b]["ours"]["ppo_s0"]["score"], "ppo_s2": ev[b]["ours"]["ppo_s2"]["score"],
            "reinforce_s0": ev[b]["ours"]["reinforce_s0"]["score"], "reinforce_s2": ev[b]["ours"]["reinforce_s2"]["score"],
            "random": ev[b]["ours"]["random"]["score"], "main_score": ev[b]["ours"]["main_score"],
            "original_ppo_s0": ev[b]["original_scores"]["ppo_s0"], "original_ppo_s2": ev[b]["original_scores"]["ppo_s2"]}
        for b in ("budget_0", "budget_10", "budget_25", "budget_50")}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
