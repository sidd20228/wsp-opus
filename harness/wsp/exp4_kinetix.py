"""Score every Exp 4 run's edited_actor.npz on Kinetix (1,024 sampled test episodes + the argmax episode).

Writes <run>/kinetix.json = {"seed", "episodes", "test": <evaluate() result>}. Skips runs that already have one.
Seed 1 for every network: the same episode draws Astra's released edit was checked on in Phase 0 (1,024/1,024).
  .venv/bin/python wsp/exp4_kinetix.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.envs.kinetix_eval import app, evaluate, load_actor  # noqa: E402

SEED, EPISODES = 1, 1024

if __name__ == "__main__":
    todo = [d for d in sorted(glob.glob(os.path.join(ROOT, "runs", "opus55", "exp4_fix_car", "*")))
            if os.path.exists(os.path.join(d, "edited_actor.npz")) and not os.path.exists(os.path.join(d, "kinetix.json"))]
    print(f"{len(todo)} networks to score")
    if todo:
        with app.run():
            for d in todo:
                res = evaluate(load_actor(os.path.join(d, "edited_actor.npz")), episodes=EPISODES, seed=SEED,
                               mode="sample")
                json.dump({"seed": SEED, "episodes": EPISODES, "test": res},
                          open(os.path.join(d, "kinetix.json"), "w"), indent=1)
                print(os.path.basename(d), "success", res["success"], "red", res["red"], "timeout", res["timeout"],
                      "argmax", res.get("argmax_outcome"), flush=True)
