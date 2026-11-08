"""Opponent trainers for Experiment 6: small 243 -> 32 -> 32 -> 82 ReLU policies, self-play against an opponent pool.

The original training code is not released (README: "PPO ... 20,000 iterations of 1,024 games"; ref.json of the
same-size runs mentions an "opponent pool"). Our choices, documented in wsp/go/README.md:

  * Each iteration plays B = 1,024 games: the learner against an opponent drawn PER GAME from a pool of frozen
    snapshots of itself (pool of 32, a snapshot every 50 iterations; with probability 0.5 the opponent is the
    learner's current weights, i.e. plain self-play). The learner is Black in half the games. Same rules, input,
    masking and temperature-1 sampling as the evaluation (wsp/go/core.py).
  * Reward +1 / -1 at the end of the game (the pgx result, so superko -> mover loses), gamma = 1.
  * PPO: shared trunk with a policy head (exported) and a value head (32 -> 1, dropped at export); GAE lambda 0.95;
    clip 0.2; 4 epochs x 4 minibatches; Adam lr 3e-4 (linearly annealed to 0), eps 1e-5; value coef 0.5;
    entropy coef 0.01; global grad-norm clip 0.5; advantages normalised per minibatch.
  * REINFORCE: same rollouts, one Adam step per iteration on -mean(logp * (R - mean R)) - 0.01 * entropy,
    lr 1e-3 (constant), grad-norm clip 0.5; no critic.
  Several seeds train in one program (vmap over seeds).
"""
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import core as C

HID = 32
POOL = 32


def init_params(key, critic="shared", critic_hid=HID):
    """critic "shared": value head on the policy trunk; "separate": its own 243-h-h-1 ReLU network (as an
    actor/critic pair; the released same-size PPO checkpoints have an "actor/..." tree, i.e. a separate critic)."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    k5, k6 = jax.random.split(jax.random.fold_in(key, 7), 2)
    def dense(k, i, o, scale):
        return (jax.nn.initializers.orthogonal(scale)(k, (i, o), jnp.float32), jnp.zeros(o, jnp.float32))
    p = {"L1": dense(k1, 3 * C.NPT, HID, np.sqrt(2)), "L2": dense(k2, HID, HID, np.sqrt(2)),
         "out": dense(k3, HID, C.NACT, 0.01)}
    if critic == "separate":
        p.update({"c1": dense(k5, 3 * C.NPT, critic_hid, np.sqrt(2)), "c2": dense(k6, critic_hid, critic_hid, np.sqrt(2)),
                  "v": dense(k4, critic_hid, 1, 1.0)})
    else:
        p["v"] = dense(k4, HID, 1, 1.0)
    return p


def policy_layers(p):
    return [p["L1"], p["L2"], p["out"]]


def forward(p, x):
    h = jax.nn.relu(x @ p["L1"][0] + p["L1"][1])
    h = jax.nn.relu(h @ p["L2"][0] + p["L2"][1])
    if "c1" in p:
        c = jax.nn.relu(x @ p["c1"][0] + p["c1"][1])
        c = jax.nn.relu(c @ p["c2"][0] + p["c2"][1])
    else:
        c = h
    return h @ p["out"][0] + p["out"][1], (c @ p["v"][0] + p["v"][1])[..., 0]


def forward_pergame(p, x):
    """p with a leading game axis on every tensor; x (G, 243)."""
    h = jax.nn.relu(jnp.einsum("gi,gij->gj", x, p["L1"][0]) + p["L1"][1])
    h = jax.nn.relu(jnp.einsum("gi,gij->gj", h, p["L2"][0]) + p["L2"][1])
    return jnp.einsum("gi,gij->gj", h, p["out"][0]) + p["out"][1]


DEFAULT = {"lam": 0.95, "lr": 3e-4, "ent": 0.01, "epochs": 4, "nmb": 4, "self_frac": 0.5, "snap_every": 50,
           "reinforce_lr": 1e-3, "anneal": True, "reward": "win", "both": False, "critic": "shared",
           "critic_hid": HID}


def reward_black(x, kind):
    """Terminal reward from Black's view: "win" = the pgx result (+1/-1; superko -> mover loses), "margin" =
    clip((black area - komi - white area) / 40, -1, 1), "mix" = average of the two. A superko ending always
    gives the pgx result."""
    R = jax.vmap(C.GAME.rewards)(x)[:, 0]
    if kind == "win":
        return R
    sc = C.scores(x).astype(jnp.float32)
    m = jnp.clip((sc[:, 0] - C.KOMI - sc[:, 1]) / 40.0, -1.0, 1.0)
    r = m if kind == "margin" else 0.5 * (R + m)
    return jnp.where(x.is_psk, R, r)


def rollout(params, pool, key, B, self_frac=0.5, reward="win", both=False):
    """Learner (params) vs per-game pool opponents. Returns per-ply learner data (T, B, ...) and per-game results."""
    k_opp, k_self, k_play = jax.random.split(key, 3)
    opp_idx = jax.random.randint(k_opp, (B,), 0, POOL)
    use_self = jax.random.bernoulli(k_self, self_frac, (B,))
    opp = jax.tree_util.tree_map(lambda a: a[opp_idx], pool)
    learner_black = jnp.arange(B) < B // 2
    x, legal = C.init_batch(B)
    done = jnp.zeros(B, bool)

    def body(carry, _):
        x, legal, done, key = carry
        f = jax.vmap(C.features)(x)
        la, v = forward(params, f)
        lo = forward_pergame(opp, f)
        lo = jnp.where(use_self[:, None], la, lo)
        me = (x.color == 0) == learner_black
        logits = jnp.where(me[:, None], la, lo)
        masked = jnp.where(legal, logits, -jnp.inf)
        key, k = jax.random.split(key)
        a = jax.random.categorical(k, masked, axis=-1)
        logp_all = jax.nn.log_softmax(jnp.where(legal, la, -jnp.inf), axis=-1)
        logp = jnp.take_along_axis(logp_all, a[:, None], 1)[:, 0]
        x2 = jax.vmap(C.GAME.step)(x, a)
        legal2 = jax.vmap(C.GAME.legal_action_mask)(x2)
        term2 = jax.vmap(C.GAME.is_terminal)(x2)
        live = ~done
        mine = live & (me | (use_self & both))
        out = {"obs": f.astype(jnp.uint8), "legal": legal, "a": a.astype(jnp.int32), "logp": logp, "v": v,
               "mine": mine, "black": x.color == 0}
        x = C._select_tree(live, x2, x)
        legal = jnp.where(live[:, None], legal2, legal)
        return (x, legal, done | term2, key), out

    (x, legal, done, _), traj = jax.lax.scan(body, (x, legal, done, k_play), None, length=C.MAX_PLY)
    rb = reward_black(x, reward)
    win = jnp.where(learner_black, jax.vmap(C.GAME.rewards)(x)[:, 0], jax.vmap(C.GAME.rewards)(x)[:, 1])
    return traj, rb, win


def gae_both(traj, rb, lam):
    """GAE separately over Black's and White's training samples (disjoint), terminal reward rb / -rb."""
    adv = 0.0
    for side, r in ((True, rb), (False, -rb)):
        t = {"mine": traj["mine"] & (traj["black"] == side), "v": traj["v"]}
        a, _ = gae(t, r, lam)
        adv = adv + a
    return adv, adv + traj["v"]


def gae(traj, R, lam=0.95):
    """Advantages over the learner's own moves (gamma = 1, terminal reward R)."""
    def body(carry, t):
        next_v, g, pending = carry
        m, v = t["mine"], t["v"]
        delta = pending + next_v - v
        g_new = delta + lam * g
        return (jnp.where(m, v, next_v), jnp.where(m, g_new, g), jnp.where(m, 0.0, pending)), jnp.where(m, g_new, 0.0)
    B = R.shape[0]
    _, adv = jax.lax.scan(body, (jnp.zeros(B), jnp.zeros(B), R), {"mine": traj["mine"], "v": traj["v"]},
                          reverse=True)
    return adv, adv + traj["v"]


def ppo_loss(p, mb, clip=0.2, vf=0.5, ent=0.01):
    logits, v = forward(p, mb["obs"].astype(jnp.float32))
    logp_all = jax.nn.log_softmax(jnp.where(mb["legal"], logits, -1e9), axis=-1)
    logp = jnp.take_along_axis(logp_all, mb["a"][..., None], -1)[..., 0]
    w = mb["mine"].astype(jnp.float32)
    n = jnp.maximum(w.sum(), 1.0)
    adv = mb["adv"]
    mu = (adv * w).sum() / n
    sd = jnp.sqrt((((adv - mu) ** 2) * w).sum() / n) + 1e-8
    adv = (adv - mu) / sd
    ratio = jnp.exp(logp - mb["logp"])
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv)
    probs = jnp.exp(logp_all)
    entropy = -(jnp.where(mb["legal"], probs * logp_all, 0.0)).sum(-1)
    vl = 0.5 * (v - mb["ret"]) ** 2
    loss = ((pg + vf * vl - ent * entropy) * w).sum() / n
    return loss, {"pg": (pg * w).sum() / n, "v": (vl * w).sum() / n, "ent": (entropy * w).sum() / n,
                  "kl": (((ratio - 1) - (logp - mb["logp"])) * w).sum() / n}


def reinforce_loss(p, mb, ent=0.01):
    logits, _ = forward(p, mb["obs"].astype(jnp.float32))
    logp_all = jax.nn.log_softmax(jnp.where(mb["legal"], logits, -1e9), axis=-1)
    logp = jnp.take_along_axis(logp_all, mb["a"][..., None], -1)[..., 0]
    w = mb["mine"].astype(jnp.float32)
    n = jnp.maximum(w.sum(), 1.0)
    probs = jnp.exp(logp_all)
    entropy = -(jnp.where(mb["legal"], probs * logp_all, 0.0)).sum(-1)
    loss = ((-logp * mb["adv"] - ent * entropy) * w).sum() / n
    return loss, {"pg": loss, "v": 0.0, "ent": (entropy * w).sum() / n, "kl": 0.0}


def make_opt(algo, iters, cfg=DEFAULT):
    if algo == "ppo":
        lr = cfg["lr"]
        sched = optax.linear_schedule(lr, 0.0, iters * cfg["epochs"] * cfg["nmb"]) if cfg["anneal"] else lr
        return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched, eps=1e-5))
    return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg["reinforce_lr"]))


def make_iteration(algo, B, iters, cfg=DEFAULT):
    cfg = dict(DEFAULT, **cfg)
    epochs, nmb = cfg["epochs"], cfg["nmb"]
    opt = make_opt(algo, iters, cfg)

    def iteration(state, _=None):
        params, opt_state, pool, key, it = state["params"], state["opt"], state["pool"], state["key"], state["it"]
        key, kr, kp = jax.random.split(key, 3)
        traj, rb, win = rollout(params, pool, kr, B, cfg["self_frac"], cfg["reward"], cfg["both"])
        if algo == "ppo":
            adv, ret = gae_both(traj, rb, cfg["lam"])
            data = dict(traj, adv=adv, ret=ret)
            data.pop("v")
            data.pop("black")
            def epoch(carry, k):
                params, opt_state = carry
                perm = jax.random.permutation(k, B).reshape(nmb, B // nmb)
                def mb_step(carry, idx):
                    params, opt_state = carry
                    mb = jax.tree_util.tree_map(lambda a: a[:, idx], data)
                    (l, info), g = jax.value_and_grad(partial(ppo_loss, ent=cfg["ent"]), has_aux=True)(params, mb)
                    upd, opt_state = opt.update(g, opt_state, params)
                    return (optax.apply_updates(params, upd), opt_state), info
                return jax.lax.scan(mb_step, (params, opt_state), perm)
            (params, opt_state), info = jax.lax.scan(epoch, (params, opt_state), jax.random.split(kp, epochs))
            info = jax.tree_util.tree_map(lambda a: a.mean(), info)
        else:
            G = jnp.where(traj["black"], rb[None, :], -rb[None, :])
            w = traj["mine"].astype(jnp.float32)
            adv = G - (G * w).sum() / jnp.maximum(w.sum(), 1.0)
            data = dict(traj, adv=adv)
            data.pop("black")
            (l, info), g = jax.value_and_grad(partial(reinforce_loss, ent=cfg["ent"]), has_aux=True)(params, data)
            upd, opt_state = opt.update(g, opt_state, params)
            params = optax.apply_updates(params, upd)
        it = it + 1
        # snapshot into the pool every 50 iterations (ring buffer)
        se = cfg["snap_every"]
        slot = (it // se) % POOL
        pool = jax.lax.cond(it % se == 0,
                            lambda: jax.tree_util.tree_map(lambda P, a: P.at[slot].set(a), pool, params),
                            lambda: pool)
        info = dict(info, win=(win > 0).mean(), length=traj["mine"].sum(0).mean() * 2.0)
        return {"params": params, "opt": opt_state, "pool": pool, "key": key, "it": it}, info

    return opt, iteration


def init_state(algo, seed, iters, cfg=DEFAULT):
    opt = make_opt(algo, iters, dict(DEFAULT, **cfg))
    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    cfg = dict(DEFAULT, **cfg)
    params = init_params(k, cfg["critic"], cfg["critic_hid"])
    pool = jax.tree_util.tree_map(lambda a: jnp.broadcast_to(a, (POOL,) + a.shape), params)
    return {"params": params, "opt": opt.init(params), "pool": pool, "key": key, "it": jnp.int32(0)}


def export(params):
    p = jax.tree_util.tree_map(np.asarray, params)
    return {"L1.W": p["L1"][0], "L1.b": p["L1"][1], "L2.W": p["L2"][0], "L2.b": p["L2"][1],
            "out.W": p["out"][0], "out.b": p["out"][1]}


def train(algo, seeds, iters, out_dir, B=1024, chunk=50, ckpt_every=500, log=print, time_limit=None, cfg=None):
    """Train len(seeds) networks in one vmapped program. Resumes from out_dir/state.pkl. Saves
    out_dir/s<seed>/iter<k>.npz every ckpt_every iterations and the final networks."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(DEFAULT, **(cfg or {}))
    _, iteration = make_iteration(algo, B, iters, cfg)

    @jax.jit
    def run_chunk(state):
        def one(st):
            return jax.lax.scan(iteration, st, None, length=chunk)
        return jax.vmap(one)(state)

    sp = os.path.join(out_dir, "state.pkl")
    if os.path.exists(sp):
        state = jax.tree_util.tree_map(jnp.asarray, pickle.load(open(sp, "rb")))
        log(f"resumed at iteration {int(state['it'][0])}")
    else:
        states = [init_state(algo, s, iters, cfg) for s in seeds]
        state = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *states)
    t0 = time.time()
    hist_path = os.path.join(out_dir, "log.jsonl")
    import json
    while int(state["it"][0]) < iters:
        t = time.time()
        state, info = run_chunk(state)
        it = int(state["it"][0])
        info = jax.tree_util.tree_map(lambda a: np.asarray(a).mean(axis=1), info)   # (seeds,)
        rec = {"it": it, "sec_per_iter": (time.time() - t) / chunk, "wall": round(time.time() - t0, 1),
               **{k: [round(float(v), 4) for v in vals] for k, vals in info.items()}}
        with open(hist_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if it % 500 == 0 or it == chunk:
            log(json.dumps(rec))
        if it % ckpt_every == 0 or it >= iters:
            host = jax.tree_util.tree_map(np.asarray, state)
            pickle.dump(host, open(sp + ".tmp", "wb"))
            os.replace(sp + ".tmp", sp)
            for i, s in enumerate(seeds):
                d = os.path.join(out_dir, f"s{s}")
                os.makedirs(d, exist_ok=True)
                np.savez(os.path.join(d, f"iter{it}.npz"),
                         **export(jax.tree_util.tree_map(lambda a: a[i], host["params"])))
            yield it
        if time_limit and time.time() - t0 > time_limit:
            log("time limit reached; stopping (resume later)")
            return
