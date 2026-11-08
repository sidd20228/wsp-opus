"""Copy trained opponents from the Modal volume `wsp-go` (train/<algo>/s<seed>/iter<k>.npz) into wsp/go/opponents/<name>/
(one .npy per tensor + arch.json + info.json) and back into the volume under opponents/<name>/.

  .venv/bin/python -m wsp.go.export_opponents
The chosen runs (see README.md, "Opponents"): PPO = train/ppo_sc (separate critic) at iteration 20,000;
REINFORCE = train/reinforce at iteration 5,000 (calibrated: practice "weak" ~95 % vs random, as the prompt says).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPP = os.path.join(HERE, "opponents")
ROLES = {"ppo_s0": "hidden", "ppo_s2": "hidden", "reinforce_s0": "hidden", "reinforce_s2": "hidden",
         "ppo_s1": "practice 'strong'", "reinforce_s1": "practice 'weak'"}


SOURCE = {"ppo": ("ppo_sc", 20000), "reinforce": ("reinforce", 5000)}


def main(source=SOURCE):
    tmp = tempfile.mkdtemp()
    for name, role in ROLES.items():
        algo, s = name.split("_s")
        sub, it = source[algo]
        f = os.path.join(tmp, f"{name}.npz")
        subprocess.run(["modal", "volume", "get", "--force", "wsp-go", f"train/{sub}/s{s}/iter{it}.npz", f],
                       check=True, capture_output=True)
        z = np.load(f)
        d = os.path.join(OPP, name)
        os.makedirs(d, exist_ok=True)
        for k in z.files:
            np.save(os.path.join(d, k + ".npy"), z[k].astype(np.float32))
        json.dump({"depth": 2, "width": 32}, open(os.path.join(d, "arch.json"), "w"))
        json.dump({"name": name, "role": role, "algo": algo, "seed": int(s), "iteration": it,
                   "games_per_iteration": 1024, "source": f"wsp-go:train/{sub}/s{s}/iter{it}.npz",
                   "settings": "see wsp/go/train.py and wsp/go/README.md"}, open(os.path.join(d, "info.json"), "w"),
                  indent=1)
        subprocess.run(["modal", "volume", "put", "--force", "wsp-go", d, f"opponents/{name}"], check=True,
                       capture_output=True)
        print("exported", name, role)


if __name__ == "__main__":
    main()
