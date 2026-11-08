"""Phase 0 checks that need no GPU: our scorers and edit readers on Astra's released answers must give the released
results. Writes results/phase0/offline_checks.json and prints a summary."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wsp import edits, exp4, score_simple  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "lm_understand-logs")
out = {}


def rd(p):
    return open(os.path.join(LOGS, p)).read()


# 0.1a Exp 1
ref = {d["true"]: d for d in json.load(open(os.path.join(LOGS, "exp1_which_game/eval/scores.json")))}
rows = []
for g in ("breakout", "pong", "pacman", "freeway"):
    s = score_simple.score_exp1(rd(f"exp1_which_game/runs/{g}/answer.md"))
    rows.append({"game": g, "ours": s["guess"], "ours_conf": s["confidence"], "ref": ref[g]["guess"],
                 "ref_conf": ref[g]["confidence"], "match": s["guess"] == ref[g]["guess"]
                 and s["confidence"] == ref[g]["confidence"]})
out["exp1_scorer"] = rows

# 0.1b Exp 2
ref2 = json.load(open(os.path.join(LOGS, "exp2_diffusion/eval/scores.json")))
ref2 = {d.get("true") or d.get("shape"): d for d in (ref2 if isinstance(ref2, list) else ref2.values())}
rows = []
for sh in ("circle", "square", "triangle"):
    s = score_simple.score_exp2(rd(f"exp2_diffusion/runs/{sh}/answer.md"))
    r = ref2.get(sh, {})
    rows.append({"shape": sh, "ours": s["guess"], "ours_conf": s["confidence"], "ref": r.get("guess"),
                 "ref_conf": r.get("confidence"), "example_filled": s["example_filled"],
                 "match": s["guess"] == r.get("guess") and s["confidence"] == r.get("confidence")})
out["exp2_scorer"] = rows

# 0.4b Exp 5: re-read Astra's replies -> the released networks
rows = []
KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
        "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
SHAPES = {"W1": (600, 32), "b1": (32,), "W2": (32, 32), "b2": (32,), "W3": (32, 6), "b3": (6,)}
for g in ("breakout", "seaquest", "asterix"):
    net = {k: np.zeros(s) for k, s in SHAPES.items()}
    rep = edits.apply(rd(f"exp5_write_from_scratch/runs/{g}/reply/answer.md"), net)
    z = np.load(os.path.join(LOGS, f"exp5_write_from_scratch/networks/{g}/hand.npz"))
    same = all(np.array_equal(net[k].astype(np.float32), z[v]) for k, v in KEYS.items())
    ref_apply = json.load(open(os.path.join(LOGS, f"exp5_write_from_scratch/runs/{g}/apply.json")))["edit_report"]
    rows.append({"game": g, "statements": rep["statements"], "applied": rep["applied"], "errors": rep["errors"][:3],
                 "ref_statements": ref_apply["statements"], "ref_applied": ref_apply["applied"],
                 "nonzero": int(sum(np.count_nonzero(v) for v in net.values())), "identical_to_released": same})
out["exp5_edits"] = rows

# 0.3b Exp 4: apply Astra's edit to the weights as shown -> 6,073 changed, equal to the released edited actor
prompt = rd("exp4_fix_car/runs/oneshot_repair/reply/prompt.txt")
net, rep, ch = exp4.apply_reply(prompt, rd("exp4_fix_car/runs/oneshot_repair/reply/answer.md"))
z = np.load(os.path.join(LOGS, "exp4_fix_car/networks/edited_actor.npz"))
maxdiff = max(float(np.max(np.abs(net[k] - z[k]))) for k in exp4.NAMES)
out["exp4_edits"] = {"statements": rep["statements"], "applied": rep["applied"], "errors": rep["errors"][:3],
                     "changed": ch["changed"], "frac_changed": ch["frac_changed"], "ref_changed": 6073,
                     "max_abs_diff_vs_released_edited": maxdiff}

os.makedirs(os.path.join(ROOT, "results/phase0"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results/phase0/offline_checks.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
