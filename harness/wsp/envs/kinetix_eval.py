"""Kinetix car-ramp evaluator for Experiment 4 (REPLICATION_PLAN.md, section 6.4 and check 0.3), run on Modal.

Scores an actor network (679 -> 128 relu -> 5 x 128 tanh -> 16 logits) on the Kinetix level `l/h2_car_ramp`,
Kinetix commit 80ee9c2, jax 0.4.38, built exactly as the official Kinetix PPO builds its env for the overrides in
`exp4_fix_car/eval/ppo_training_result.json` (env=symbolic -> multi_discrete actions + symbolic_flat observations,
env_size=l, dense_reward_scale 0.2, frame_skip 2, max_timesteps 256, level loaded with the l-size static params).

Network (as in the Exp 4 prompt, "THE NETWORK"):
    embed = relu(x @ embed.W + embed.b); a_0 = tanh(embed @ actor0.W + actor0.b); a_k = tanh(a_(k-1) @ actor<k>.W + b)
    logits = a_4 @ actor_head.W + actor_head.b; 16 logits -> 6 action entries (3,3,3,3,2,2 values), each sampled
    independently from its softmax ("sample") or its argmax ("argmax").

Episode outcome (first matching): a substep reward of +1 (green touches blue) = success, -1 (green touches red;
red wins inside a substep) = red, NaN anywhere in the state = invalid, 256 steps = timeout. Same rule as Kinetix's
`engine_step` `done`. "progress" = 1 for a success, else (d0 - min d) / d0 with d the green-blue centre distance
(our reading of the logs' "mean progress toward the goal"; it reproduces the base network's 0.339).

CLI (from the project root):
    .venv/bin/modal run wsp/envs/kinetix_eval.py --npz <path> --episodes 1024 --seed 0 --mode sample
    .venv/bin/modal run wsp/envs/kinetix_eval.py --check-obs     # observation layout vs the prompt
    .venv/bin/modal run wsp/envs/kinetix_eval.py --verify        # check 0.3 -> results/phase0/kinetix_check.json

Python (the harness):
    from wsp.envs.kinetix_eval import evaluate, load_actor, app
    params = load_actor("edited_actor.npz")          # or a dict {"embed.W": ..., ..., "actor_head.b": ...}
    res = evaluate(params, episodes=1024, seed=1, mode="sample")   # starts the Modal app for this call
    # several calls share one container (and its compile cache) if wrapped in `with app.run(): ...`

Set WSP_KINETIX_GPU=cpu (or e.g. A10G) before import to change the hardware; the default is an L4 (A10G if no L4 is free).
"""
import contextlib
import json
import os
import pathlib
import time

import modal
import numpy as np

KINETIX_COMMIT = "80ee9c292837c825cafaf9889323fdcc212efd1b"
JAX_VERSION = "0.4.38"
LEVEL = "l/h2_car_ramp"
MAX_STEPS = 256
TENSORS = ["embed.W", "embed.b"] + [f"actor{k}.{t}" for k in range(5) for t in "Wb"] + ["actor_head.W", "actor_head.b"]
SHAPES = {"embed.W": (679, 128), "embed.b": (128,), "actor_head.W": (128, 16), "actor_head.b": (16,),
          **{f"actor{k}.W": (128, 128) for k in range(5)}, **{f"actor{k}.b": (128,) for k in range(5)}}
ENTRY_SLICES = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 14), (14, 16)]
OUTCOMES = {1: "success", 2: "red", 3: "invalid", 4: "timeout"}

# PPO checkpoint names (Kinetix ActorCriticSymbolicRNN) -> the prompt's names.
PPO_NAMES = {"params/Dense_0": "embed", "params/GeneralActorCriticRNN_0/Dense_0": "actor_head",
             **{f"params/GeneralActorCriticRNN_0/DenseAndActivation_{k}/Dense_0": f"actor{k}" for k in range(5)}}


# ----------------------------------------------------------------------------------------------------------------
# Loading (local, numpy only)
# ----------------------------------------------------------------------------------------------------------------
def to_actor_dict(arrays) -> dict:
    """Map a dict/npz of arrays in either format (edited names, or Kinetix PPO checkpoint names) to the edited
    format: {"embed.W", "embed.b", "actor0.W".."actor4.b", "actor_head.W", "actor_head.b"} as float32 numpy."""
    keys = set(arrays.keys())
    out = {}
    if "embed.W" in keys:
        for t in TENSORS:
            out[t] = np.asarray(arrays[t], dtype=np.float32)
    elif "params/Dense_0/kernel" in keys:
        for src, dst in PPO_NAMES.items():
            out[dst + ".W"] = np.asarray(arrays[src + "/kernel"], dtype=np.float32)
            out[dst + ".b"] = np.asarray(arrays[src + "/bias"], dtype=np.float32)
    else:
        raise ValueError(f"unknown actor format; keys: {sorted(keys)[:6]} ...")
    for t in TENSORS:
        if out[t].shape != SHAPES[t]:
            raise ValueError(f"{t}: shape {out[t].shape}, expected {SHAPES[t]}")
        if not np.all(np.isfinite(out[t])):
            raise ValueError(f"{t}: non-finite entries")
    return out


def load_actor(path) -> dict:
    with np.load(path) as z:
        return to_actor_dict({k: z[k] for k in z.files})


def forward_numpy(params: dict, obs: np.ndarray) -> np.ndarray:
    """Reference forward pass (float64) for checks."""
    p = {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}
    h = np.maximum(obs @ p["embed.W"] + p["embed.b"], 0.0)
    for k in range(5):
        h = np.tanh(h @ p[f"actor{k}.W"] + p[f"actor{k}.b"])
    return h @ p["actor_head.W"] + p["actor_head.b"]


# ----------------------------------------------------------------------------------------------------------------
# Modal app
# ----------------------------------------------------------------------------------------------------------------
_HW = os.environ.get("WSP_KINETIX_GPU", "L4,A10G")       # comma list = fallback order
GPU = None if _HW.lower() in ("cpu", "none", "") else _HW.split(",")
JAX_PKG = f"jax[cuda12]=={JAX_VERSION}" if GPU else f"jax=={JAX_VERSION}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(JAX_PKG, f"jaxlib=={JAX_VERSION}",
                 f"kinetix-env @ git+https://github.com/FLAIROx/Kinetix.git@{KINETIX_COMMIT}")
    .env({"XLA_PYTHON_CLIENT_PREALLOCATE": "false", "WANDB_MODE": "disabled", "SDL_VIDEODRIVER": "dummy",
          "PYGAME_HIDE_SUPPORT_PROMPT": "1"})
)
if __name__.startswith("wsp."):
    # Imported as a package module (the harness): the container must be able to import `wsp.envs.kinetix_eval`.
    image = image.add_local_python_source("wsp")
app = modal.App("wsp-kinetix")

_CACHE = {}


def _setup():
    """Build the env the way experiments/ppo.py does for env=symbolic env_size=l train_levels=[l/h2_car_ramp]."""
    if "env" in _CACHE:
        return _CACHE
    import jax
    import jax.numpy as jnp
    from kinetix.environment.env import KinetixEnv
    from kinetix.environment.spaces import MultiDiscreteActions, SymbolicObservations
    from kinetix.environment.ued.ued import make_reset_fn_list_of_levels
    from kinetix.environment.utils import static_env_params_from_size
    from kinetix.util.config import generate_params_from_config

    # generate_params_from_config(config) after normalise_config(): env (symbolic.yaml) + env_size l.
    _, size_cfg = static_env_params_from_size("l", return_dict=True)
    config = {"env_size_type": "predefined", "dense_reward_scale": 0.2, **size_cfg}
    env_params, static_env_params = generate_params_from_config(config)
    reset_fn = make_reset_fn_list_of_levels([LEVEL], static_env_params)

    class CarRampEnv(KinetixEnv):
        """KinetixEnv (as make_kinetix_env builds it for multi_discrete + symbolic_flat), whose info also carries
        the sparse reward of the step (+1 goal / -1 red / 0), taken at the first terminating substep by
        engine_step. Dynamics, observation and reward are Kinetix's own."""

        def compute_reward_info(self, state, manifolds):
            reward, info = super().compute_reward_info(state, manifolds)
            info["sparse"] = reward
            return reward, info

    env = CarRampEnv(
        action_type=MultiDiscreteActions(env_params, static_env_params),
        observation_type=SymbolicObservations(env_params, static_env_params),
        static_env_params=static_env_params,
        reset_function=reset_fn,
        auto_reset=False,
    )
    _CACHE.update(env=env, env_params=env_params, static_env_params=static_env_params, jax=jax, jnp=jnp)
    return _CACHE


def _make_rollout(n: int, greedy: bool, precision: str):
    key_ = ("rollout", n, greedy, precision)
    if key_ in _CACHE:
        return _CACHE[key_]
    c = _setup()
    jax, jnp, env, env_params = c["jax"], c["jnp"], c["env"], c["env_params"]

    def logits_fn(p, x):
        ctx = contextlib.nullcontext() if precision == "default" else jax.default_matmul_precision(precision)
        with ctx:
            h = jax.nn.relu(x @ p["embed.W"] + p["embed.b"])
            for k in range(5):
                h = jnp.tanh(h @ p[f"actor{k}.W"] + p[f"actor{k}.b"])
            return h @ p["actor_head.W"] + p["actor_head.b"]

    def act(logits, key):
        keys = jax.random.split(key, len(ENTRY_SLICES))
        out = []
        for (a, b), k in zip(ENTRY_SLICES, keys):
            lg = logits[a:b]
            out.append(jnp.argmax(lg) if greedy else jax.random.categorical(k, lg))
        return jnp.stack(out).astype(jnp.int32)

    def goal_distance(state):
        # Distance between the green and the blue shapes' centres (the dense-reward distance).
        pos = jnp.concatenate([state.polygon.position, state.circle.position], 0)
        roles = jnp.concatenate([state.polygon_shape_roles * state.polygon.active,
                                 state.circle_shape_roles * state.circle.active], 0)
        d = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        return jnp.sum(jnp.where((roles[:, None] == 1) & (roles[None, :] == 2), d, 0.0))

    def is_nan(state):
        return jax.tree.reduce(jnp.logical_or, jax.tree.map(lambda x: jnp.isnan(x).any(), state), False)

    def trace_row(state, action, probs0):
        return jnp.concatenate([
            state.polygon.position[5], state.polygon.rotation[5:6],          # car body (shape 5)
            state.circle.position[0], state.circle.position[1],              # green wheel (12), wheel (13)
            action[:1].astype(jnp.float32), probs0])

    def init_one(key):
        obs, state = env.reset(key, env_params)
        d0 = goal_distance(state)
        return dict(obs=obs, state=state, active=jnp.array(True), outcome=jnp.array(0, jnp.int32),
                    ret=jnp.array(0.0, jnp.float32), length=jnp.array(0, jnp.int32), d0=d0, dmin=d0)

    def step_one(p, c, key):
        k_act, k_env = jax.random.split(key)
        logits = logits_fn(p, c["obs"])
        action = act(logits, k_act)
        obs, state, reward, done, info = env.step_env(k_env, c["state"], action, env_params)
        nan = is_nan(state)
        timeout = state.timestep >= env_params.max_timesteps
        code = jnp.where(info["sparse"] > 0, 1, jnp.where(info["sparse"] < 0, 2, jnp.where(nan, 3, 4)))
        a = c["active"]
        new = dict(
            obs=obs, state=state,
            active=a & ~done,
            outcome=jnp.where(a & done, code, c["outcome"]).astype(jnp.int32),
            ret=c["ret"] + jnp.where(a, reward, 0.0),
            length=c["length"] + a.astype(jnp.int32),
            d0=c["d0"],
            dmin=jnp.where(a & ~nan, jnp.minimum(c["dmin"], info["distance"]), c["dmin"]),
        )
        row = trace_row(c["state"], action, jax.nn.softmax(logits[0:3]))
        consistent = ~a | (done == ((info["sparse"] != 0) | nan | timeout))
        return new, row, consistent

    @jax.jit
    def rollout(p, key):
        k_reset, k_run = jax.random.split(key)
        carry = jax.vmap(init_one)(jax.random.split(k_reset, n))
        trace = jnp.full((MAX_STEPS, 11), jnp.nan, jnp.float32)

        def cond(s):
            t, carry, _, _, _ = s
            return (t < MAX_STEPS) & carry["active"].any()

        def body(s):
            t, carry, trace, ok, k = s
            k, ks = jax.random.split(k)
            carry, rows, cons = jax.vmap(step_one, in_axes=(None, 0, 0))(p, carry, jax.random.split(ks, n))
            trace = trace.at[t].set(jnp.where(carry["length"][0] > t, rows[0], jnp.nan))
            return t + 1, carry, trace, ok & cons.all(), k

        t, carry, trace, ok, _ = jax.lax.while_loop(cond, body, (0, carry, trace, jnp.array(True), k_run))
        prog = jnp.where(carry["outcome"] == 1, 1.0,
                         jnp.clip((carry["d0"] - carry["dmin"]) / carry["d0"], 0.0, 1.0))
        return dict(outcome=carry["outcome"], ret=carry["ret"], length=carry["length"], progress=prog,
                    trace=trace, consistent=ok, steps=t)

    _CACHE[key_] = rollout
    return rollout


def _summarise(r, n):
    out = {"episodes": int(n)}
    oc = np.asarray(r["outcome"])
    for code, name in OUTCOMES.items():
        out[name] = float(np.mean(oc == code))
    out["unfinished"] = int(np.sum(oc == 0))
    out["return"] = float(np.mean(np.asarray(r["ret"], np.float64)))
    out["length"] = float(np.mean(np.asarray(r["length"], np.float64)))
    out["progress"] = float(np.mean(np.asarray(r["progress"], np.float64)))
    out["success_count"] = int(np.sum(oc == 1))
    out["kinetix_done_consistent"] = bool(r["consistent"])
    return out


def _trace_rows(r, every=16):
    tr = np.asarray(r["trace"])
    L = int(np.asarray(r["length"])[0])
    steps = sorted(set(list(range(0, L, every)) + [L - 1]))
    rows = []
    for s in steps:
        v = tr[s]
        rows.append({"step": s, "body": [round(float(x), 3) for x in v[0:3]],
                     "green": [round(float(x), 3) for x in v[3:5]], "wheel": [round(float(x), 3) for x in v[5:7]],
                     "action0": int(v[7]), "p0": [round(float(x), 3) for x in v[8:11]]})
    return {"outcome": OUTCOMES.get(int(np.asarray(r["outcome"])[0]), "unfinished"), "length": L, "rows": rows}


@app.function(image=image, gpu=GPU, timeout=3600, scaledown_window=60)
def run_eval(params: dict, episodes: int = 1024, seed: int = 0, mode: str = "sample", argmax: bool = True,
             precision: str = "default", trace: bool = True) -> dict:
    """Score an actor (edited-format dict of float32 arrays) on `episodes` episodes. Returns shares/means plus the
    argmax-mode episode outcome."""
    c = _setup()
    jax, jnp = c["jax"], c["jnp"]
    p = {k: jnp.asarray(np.asarray(v, np.float32)) for k, v in to_actor_dict(params).items()}
    res = {"level": LEVEL, "kinetix_commit": KINETIX_COMMIT, "jax": jax.__version__,
           "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind, "seed": int(seed), "mode": mode, "precision": precision}

    t0 = time.time()
    fn = _make_rollout(int(episodes), mode == "argmax", precision)
    r = jax.device_get(fn(p, jax.random.PRNGKey(seed)))
    res["wall_s"] = round(time.time() - t0, 2)
    res.update(_summarise(r, episodes))
    if trace:
        res["episode0"] = _trace_rows(r)

    if argmax:
        t0 = time.time()
        fa = _make_rollout(1, True, precision)
        ra = jax.device_get(fa(p, jax.random.PRNGKey(seed)))
        res["argmax_wall_s"] = round(time.time() - t0, 2)
        res["argmax_outcome"] = OUTCOMES.get(int(ra["outcome"][0]), "unfinished")
        res["argmax_success"] = float(ra["outcome"][0] == 1)
        res["argmax_return"] = float(ra["ret"][0])
        res["argmax_length"] = int(ra["length"][0])
        if trace:
            res["argmax_episode"] = _trace_rows(ra)
    return res


@app.function(image=image, gpu=GPU, timeout=1800, scaledown_window=60)
def initial_observation() -> dict:
    """The first observation of the level and the decoded fields, for the layout check."""
    c = _setup()
    jax = c["jax"]
    obs, state = c["env"].reset(jax.random.PRNGKey(0), c["env_params"])
    sp = c["static_env_params"]
    return {"obs": np.asarray(obs).tolist(), "obs_dim": int(obs.shape[0]),
            "static_env_params": {k: getattr(sp, k) for k in ["num_polygons", "num_circles", "num_joints",
                                                              "num_thrusters", "frame_skip", "num_motor_bindings",
                                                              "num_thruster_bindings", "num_static_fixated_polys"]},
            "env_params": {"max_timesteps": int(c["env_params"].max_timesteps),
                           "dense_reward_scale": float(c["env_params"].dense_reward_scale),
                           "dt": float(c["env_params"].dt)},
            "gravity": np.asarray(state.gravity).tolist(),
            "motor_bindings": np.asarray(state.motor_bindings).tolist(),
            "motor_auto": np.asarray(state.motor_auto).tolist()}


# ----------------------------------------------------------------------------------------------------------------
# Observation layout check against the prompt ("THE SCENE AND THE RULES")
# ----------------------------------------------------------------------------------------------------------------
def layout_checks(obs) -> list:
    o = np.asarray(obs, np.float64)
    checks = []

    def chk(name, got, exp, tol=2e-3):
        got = np.atleast_1d(np.asarray(got, np.float64))
        exp = np.atleast_1d(np.asarray(exp, np.float64))
        checks.append({"check": name, "expected": exp.round(4).tolist(), "got": got.round(4).tolist(),
                       "ok": bool(got.shape == exp.shape and np.allclose(got, exp, atol=tol))})

    chk("length 679", [len(o)], [679])
    chk("entry 678 = gravity y / 10", o[678], -0.981)
    poly = {0: 0, 4: 26, 5: 52, 6: 78, 7: 104}
    # (centre, role one-hot, static, sin, cos)
    scene = {0: ((2.5, -4.8), (1, 0, 0, 0), True, 0.0), 4: ((0.232, 0.762), (0, 0, 1, 0), False, 0.0),
             5: ((2.8, 0.725), (1, 0, 0, 0), False, 0.0), 6: ((2.227, 0.558), (1, 0, 0, 0), True, 0.95),
             7: ((1.622, 0.402), (0, 0, 0, 1), True, 0.0)}
    for sid, (centre, role, static, ang) in scene.items():
        b = poly[sid]
        chk(f"poly slot {sid}: position", o[b:b + 2], centre)
        chk(f"poly slot {sid}: velocity 0", o[b + 2:b + 4], (0, 0))
        if static:
            chk(f"poly slot {sid}: static (inv mass, inv inertia 0)", o[b + 4:b + 6], (0, 0))
        else:
            checks.append({"check": f"poly slot {sid}: dynamic (inv mass > 0)", "got": [round(o[b + 4], 4)],
                           "expected": "> 0", "ok": bool(o[b + 4] > 0)})
        chk(f"poly slot {sid}: density 1, tanh(angvel/10) 0", o[b + 6:b + 8], (1, 0))
        chk(f"poly slot {sid}: role one-hot", o[b + 8:b + 12], role)
        chk(f"poly slot {sid}: sin, cos", o[b + 12:b + 14], (np.sin(ang), np.cos(ang)))
        chk(f"poly slot {sid}: friction 1, restitution 0, 0 (=polygon)", o[b + 14:b + 17], (1, 0, 0))
        chk(f"poly slot {sid}: <=3 vertices flag 0", o[b + 25], 0)
    chk("poly slot 0: vertices |x|,|y| = 2.5, 5.2", np.abs(o[17:25]).reshape(4, 2), [[2.5, 5.2]] * 4)
    chk("poly slots 8..11 empty", o[130:234], np.zeros(104))
    circ = {12: (234, (2.537, 0.735), (0, 1, 0, 0), 0.22), 13: (252, (3.107, 0.75), (1, 0, 0, 0), 0.229)}
    for sid, (b, centre, role, rad) in circ.items():
        chk(f"circle slot {sid}: position", o[b:b + 2], centre)
        chk(f"circle slot {sid}: role one-hot", o[b + 8:b + 12], role)
        chk(f"circle slot {sid}: friction 1, restitution 0", o[b + 14:b + 16], (1, 0))
        chk(f"circle slot {sid}: radius, 1 (=circle)", o[b + 16:b + 18], (rad, 1))
    chk("circle slots 14, 15 empty", o[270:306], np.zeros(36))
    joints = {0: (306, (0, 0), (0.307, 0.025), 13), 1: (360, (0, 0), (-0.262, 0.01), 12)}
    for j, (b, pb, pa, sb) in joints.items():
        chk(f"joint {j}: active 1, fixed 0", o[b:b + 2], (1, 0))
        chk(f"joint {j}: position on b", o[b + 2:b + 4], pb)
        chk(f"joint {j}: position on a", o[b + 4:b + 6], pa)
        chk(f"joint {j}: speed 3, power 1, motor on 1, limits 0", o[b + 8:b + 12], (3, 1, 1, 0))
        chk(f"joint {j}: binding one-hot = 0", o[b + 12:b + 16], (1, 0, 0, 0))
        chk(f"joint {j}: limit terms 0", o[b + 16:b + 22], np.zeros(6))
        chk(f"joint {j}: shape b one-hot = {sb}", o[b + 22:b + 38], np.eye(16)[sb])
        chk(f"joint {j}: shape a one-hot = 5", o[b + 38:b + 54], np.eye(16)[5])
    chk("joint slots 2..5 empty", o[414:630], np.zeros(216))
    chk("thruster slots empty", o[630:678], np.zeros(48))
    return checks


# ----------------------------------------------------------------------------------------------------------------
# Python entry point for the harness
# ----------------------------------------------------------------------------------------------------------------
def evaluate(params, episodes: int = 1024, seed: int = 0, mode: str = "sample", argmax: bool = True,
             precision: str = "default", trace: bool = True) -> dict:
    """Evaluate an actor given as an edited-format dict of numpy arrays (or PPO-format dict, or an .npz path).
    Uses the running `wsp-kinetix` app if called inside `with app.run():`, else starts one for this call."""
    params = load_actor(params) if isinstance(params, (str, pathlib.Path)) else to_actor_dict(params)
    kw = dict(params=params, episodes=episodes, seed=seed, mode=mode, argmax=argmax, precision=precision,
              trace=trace)
    if app.app_id is not None:
        return run_eval.remote(**kw)
    with app.run():
        return run_eval.remote(**kw)


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
@app.local_entrypoint()
def main(npz: str = "", episodes: int = 1024, seed: int = 0, mode: str = "sample", precision: str = "default",
         out: str = "", check_obs: bool = False, verify: bool = False, base_seed: int = 0, edited_seed: int = 1):
    """--npz <path>: evaluate one network. --check-obs: observation layout check. --verify: check 0.3."""
    if check_obs:
        return check_observation(out)
    if verify:
        return verify_released(base_seed, edited_seed, precision)
    if not npz:
        raise SystemExit("give --npz <path>, --check-obs or --verify")
    params = load_actor(npz)
    res = run_eval.remote(params=params, episodes=episodes, seed=seed, mode=mode, precision=precision)
    res["npz"] = npz
    text = json.dumps(res, indent=1)
    print(text)
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(text)


def check_observation(out: str = ""):
    info = initial_observation.remote()
    checks = layout_checks(info["obs"])
    bad = [c for c in checks if not c["ok"]]
    for c in checks:
        print(("OK  " if c["ok"] else "BAD ") + c["check"], "" if c["ok"] else f"got {c['got']} expected {c['expected']}")
    print(f"{len(checks) - len(bad)}/{len(checks)} layout checks pass")
    print(json.dumps({k: v for k, v in info.items() if k != "obs"}, indent=1))
    if out:
        pathlib.Path(out).write_text(json.dumps({"checks": checks, **info}, indent=1))


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]   # only used locally (the container has a flat /root)


def verify_released(base_seed: int = 0, edited_seed: int = 1, precision: str = "default"):
    """Check 0.3: base network 0 % on 256 sampled episodes; edited network ~100 % on 1,024 sampled episodes."""
    LOGS = _project_root() / "lm_understand-logs" / "exp4_fix_car"
    RESULTS = _project_root() / "results" / "phase0" / "kinetix_check.json"
    base = load_actor(LOGS / "networks" / "ppo_final_seed0.npz")
    edited = load_actor(LOGS / "networks" / "edited_actor.npz")
    info = initial_observation.remote()
    checks = layout_checks(info["obs"])
    obs0 = np.asarray(info["obs"])
    # Reference forward pass at the first observation (numpy float64), for the record.
    ref = {name: {"logits": forward_numpy(p, obs0).round(4).tolist()} for name, p in [("base", base), ("edited", edited)]}

    runs = {}
    runs["base_256"] = run_eval.remote(params=base, episodes=256, seed=base_seed, precision=precision)
    runs["edited_1024"] = run_eval.remote(params=edited, episodes=1024, seed=edited_seed, precision=precision)
    runs["edited_256"] = run_eval.remote(params=edited, episodes=256, seed=base_seed, precision=precision)
    runs["base_1024"] = run_eval.remote(params=base, episodes=1024, seed=edited_seed, precision=precision)

    orig_base = json.loads((LOGS / "eval" / "base_result.json").read_text())
    orig_edit = json.loads((LOGS / "eval" / "oneshot_result.json").read_text())
    keys = ["success", "red", "timeout", "invalid", "return", "length", "progress", "argmax_outcome"]
    out = {
        "check": "0.3 Exp 4 Kinetix (REPLICATION_PLAN.md section 4)",
        "level": LEVEL, "kinetix_commit": KINETIX_COMMIT, "jax": JAX_VERSION, "hardware": "/".join(GPU) if GPU else "cpu",
        "seeds": {"base_256": base_seed, "edited_1024": edited_seed, "edited_256": base_seed,
                  "base_1024": edited_seed, "argmax": "deterministic"},
        "layout_checks_passed": f"{sum(c['ok'] for c in checks)}/{len(checks)}",
        "layout_checks_failed": [c for c in checks if not c["ok"]],
        "env": {k: v for k, v in info.items() if k != "obs"},
        "ours": runs,
        "original": {"base_256": {k: orig_base.get(k) for k in keys},
                     "edited_256_search": {k: orig_edit["search"].get(k) for k in keys},
                     "edited_1024_test": orig_edit["test"]},
        "first_obs_logits": ref,
    }
    out["pass"] = bool(runs["base_256"]["success"] == 0.0 and runs["edited_1024"]["success"] >= 0.99)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    for name, r in runs.items():
        print(f"{name:12s} success {r['success']:.4f} red {r['red']:.4f} timeout {r['timeout']:.4f} "
              f"invalid {r['invalid']:.4f} return {r['return']:.4f} length {r['length']:.2f} "
              f"progress {r['progress']:.3f} argmax {r['argmax_outcome']} wall {r['wall_s']}s")
    print("layout checks", out["layout_checks_passed"], "pass:", out["pass"], "->", RESULTS)
