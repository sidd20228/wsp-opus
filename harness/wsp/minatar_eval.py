"""Experiment 5 scorer: mean score over N episodes of the official MinAtar 1.0.15 game, actions sampled from the
softmax of the logits, sticky actions 0.1, episodes cut at 10,000 steps (exp5 README). The 6 input channels are built
from the game's own channels exactly as each prompt's "THE INPUT" section says.

  python wsp/minatar_eval.py <game> <net.npz> [--episodes 1000] [--seed 0]
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np

CHANNELS = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
}
KEYS = {"W1": "actor/dense1/kernel", "b1": "actor/dense1/bias", "W2": "actor/dense2/kernel",
        "b2": "actor/dense2/bias", "W3": "actor/logits/kernel", "b3": "actor/logits/bias"}
MAX_STEPS = 10_000


def load_net(path_or_dict):
    if isinstance(path_or_dict, dict):
        return {k: np.asarray(v, np.float64) for k, v in path_or_dict.items()}
    z = np.load(path_or_dict)
    return {k: z[v].astype(np.float64) if v in z.files else z[k].astype(np.float64) for k, v in KEYS.items()}


def _obs(state, idx_groups):
    # state: (10, 10, C) bool -> 600 numbers, input i = (row * 10 + col) * 6 + channel
    x = np.zeros((10, 10, 6))
    for c, src in enumerate(idx_groups):
        if src:
            x[:, :, c] = state[:, :, src].any(axis=2)
    return x.reshape(600)


def _episodes(args):
    game, net, seeds = args
    from minatar import Environment
    env = Environment(game, sticky_action_prob=0.1)
    names = env.env.channels
    groups = [[names[n] for n in g] for g in CHANNELS[game]]
    out = []
    for s in seeds:
        env.seed(int(s))
        rng = np.random.default_rng(int(s) + 10_000_019)
        env.reset()
        total, steps, done = 0.0, 0, False
        while not done and steps < MAX_STEPS:
            x = _obs(env.state(), groups)
            h1 = np.tanh(x @ net["W1"] + net["b1"])
            h2 = np.tanh(h1 @ net["W2"] + net["b2"])
            z = h2 @ net["W3"] + net["b3"]
            p = np.exp(z - z.max())
            p /= p.sum()
            a = int(rng.choice(6, p=p))
            r, done = env.act(a)
            total += r
            steps += 1
        out.append((total, steps))
    return out


def evaluate(game, net, episodes=1000, seed=0, workers=None):
    net = load_net(net)
    seeds = np.arange(episodes) + seed * 1_000_003
    workers = workers or min(10, os.cpu_count() or 1)
    chunks = [(game, net, c) for c in np.array_split(seeds, workers * 4) if len(c)]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = [x for part in pool.map(_episodes, chunks) for x in part]
    scores = np.array([r[0] for r in res])
    lens = np.array([r[1] for r in res])
    return {"game": game, "episodes": len(res), "mean": float(scores.mean()),
            "se": float(scores.std(ddof=1) / np.sqrt(len(res))), "mean_length": float(lens.mean()),
            "cut_at_max_steps": int((lens >= MAX_STEPS).sum()), "seed": seed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("npz")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(evaluate(a.game, a.npz, a.episodes, a.seed)))
