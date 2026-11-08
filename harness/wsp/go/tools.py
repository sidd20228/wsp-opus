"""Experiment 6 tools: the games pass, the positions pass, the hidden evaluation and head-to-head games.

All functions take the written network as a dict {name: array} (the 14 tensors; missing ones are zeros, as in the
original "tensors never saved stay zero"), or a folder of <name>.npy files (e.g. a copy of the sandbox's /work/net).

  games_pass(net, request, out_dir, k=None, budget=None)      -> str   (writes summary.json, README.txt, games.json,
                                                                         boards.npy, valid.npy into out_dir)
  positions_pass(net, request, out_dir, k=None, budget=None)  -> str   (writes README.txt, inputs.npy, legal.npy,
                                                                         logits.npy, probs.npy[, act_L1..act_L6.npy])
  pass_ok(text) -> bool        did the pass succeed (a failed request is not counted)
  evaluate(net, games=2000)    -> dict in the format of eval/budget_N/eval_submission.json
  h2h(net_a, net_b, games=2000, seed=31337) -> summary dict (net_a's view, net_a Black in half)
  analysis_strong(net, games=500, seed=4242) -> {"opponent": "strong", "summary": ...}

The returned text is exactly what the original harness sent back to the model:
  "Games pass {k} of {budget} done.\n```\n{json}\n```\nPasses left: {budget - k}."
With k=None it is inferred from out_dir's name (pass_<k>); with budget=None only the JSON line is returned.

backend="local" (default) runs JAX on this machine (CPU is fast enough: 128 recorded games < 2 s, 2,000 games ~3 s);
backend="modal" runs the same code on a Modal GPU in an ephemeral app (wsp.go.modal_app.remote_call).
"""
import io
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPP_DIR = os.environ.get("WSP_GO_OPP_DIR", os.path.join(HERE, "opponents"))
ARCH = {"depth": 6, "width": 512}

PRACTICE = {"random": "random", "strong": "ppo_s1", "weak": "reinforce_s1"}
HIDDEN = {"ppo_s0": "ppo_s0", "ppo_s2": "ppo_s2", "reinforce_s0": "reinforce_s0", "reinforce_s2": "reinforce_s2",
          "random": "random"}
# fixed seeds per practice opponent (the original's games passes reused the same random stream every pass: the
# random opponent's moves in game 0 are identical in runs budget_10 and budget_50)
PRACTICE_SEED = {"random": 101, "strong": 102, "weak": 103}

GAMES_README = ("games.json: one entry per game (opponent, net_color, result +1/-1, margin = net's area score minus the "
                "opponent's incl. komi, end, stones captured / lost, moves: ply, by, move (row*9+col, 81 = pass), and "
                "for the network's moves its top-5 [move, probability]). boards.npy: (games, 162, 81) int8, the board "
                "AFTER each ply (1 Black, -1 White, 0 empty; index row*9+col); valid.npy: (games, 162) which plies "
                "were played.")
POS_README_BASE = ("inputs.npy (P, 243) the network inputs; legal.npy (P, 82) the legal moves (for 'board' positions: "
                   "the empty points and pass; suicide and ko are not checked); logits.npy, probs.npy (P, 82); ")
POS_README_ACT = "act_L1.npy .. act_L6.npy (P, 512) the hidden activations after ReLU."
POS_README_NOACT = "no activations (ask with \"activations\": true)."


# ----------------------------------------------------------------------------------------------- networks
def shapes(arch=ARCH):
    d, w = arch["depth"], arch["width"]
    s = {}
    for k in range(1, d + 1):
        s[f"L{k}.W"] = (243 if k == 1 else w, w)
        s[f"L{k}.b"] = (w,)
    s["out.W"], s["out.b"] = (w, 82), (82,)
    return s


def written_net(net, arch=ARCH):
    """The written network as a full dict of tensors (missing -> zeros), shape- and finiteness-checked."""
    from . import core as C
    t = C.load_net(net)
    out = {}
    for name, shp in shapes(arch).items():
        a = np.asarray(t[name], np.float32) if name in t else np.zeros(shp, np.float32)
        if a.shape != shp:
            raise ValueError(f"{name}: shape {a.shape}, expected {shp}")
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: not finite")
        out[name] = a
    return out


def load_opponent(name):
    from . import core as C
    if name == "random":
        return "random"
    return C.tensors_to_layers(C.load_net(os.path.join(OPP_DIR, name)))


def _layers(net):
    from . import core as C
    if isinstance(net, str) and net == "random":
        return "random"
    t = C.load_net(net)
    return C.tensors_to_layers(t)


def _infer_k(out_dir):
    m = re.search(r"pass_(\d+)/?$", os.path.normpath(out_dir) if out_dir else "")
    return int(m.group(1)) if m else None


def _wrap(kind, k, budget, payload, ok):
    line = json.dumps(payload)
    if budget is None:
        return line
    left = budget - k if ok else budget - (k - 1)
    head = (f"{'Games' if kind == 'games' else 'Positions'} pass {k} of {budget} done." if ok else
            f"{'Games' if kind == 'games' else 'Positions'} pass request failed; it was not counted.")
    return f"{head}\n```\n{line}\n```\nPasses left: {left}."


def pass_ok(text):
    """True when a pass text reports success (the request counts as a pass)."""
    return '"ok": true' in text


def _write_files(out_dir, files):
    os.makedirs(out_dir, exist_ok=True)
    for name, data in files.items():
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)


def _npy_bytes(a):
    b = io.BytesIO()
    np.save(b, a)
    return b.getvalue()


def _remote(fn, *args, **kwargs):
    import modal
    from .modal_app import app, remote_call
    with app.run():
        return remote_call.remote(fn, *args, **kwargs)


# ----------------------------------------------------------------------------------------------- games pass
def validate_games_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("games"), dict):
        return 'the request must be JSON {"games": {"strong": n, "weak": n, "random": n}}'
    extra = set(request) - {"games"}
    if extra:
        return f"unknown keys {sorted(extra)}"
    g = request["games"]
    if not g:
        return "no games requested"
    for name, n in g.items():
        if name not in PRACTICE:
            return f"unknown opponent {name!r}; the opponents are \"random\", \"weak\" and \"strong\""
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return f"{name}: the number of games must be a non-negative integer"
        if n % 2:
            return f"{name}: the number of games must be even ({n})"
    tot = sum(g.values())
    if tot == 0:
        return "no games requested"
    if tot > 128:
        return f"at most 128 games in total ({tot} requested)"
    return None


def games_pass_core(net, request, k, budget=None, arch=ARCH):
    """-> (text, files {name: bytes}, ok)."""
    from . import core as C
    err = validate_games_request(request)
    if err is None:
        try:
            layers = C.tensors_to_layers(written_net(net, arch))
        except Exception as e:                                   # noqa: BLE001
            err = f"the network could not be loaded: {e}"
    if err is not None:
        return _wrap("games", k, budget, {"ok": False, "kind": "games", "error": err}, False), {}, False
    per_opp, games, boards, valid = {}, [], [], []
    gi = 0
    for name in sorted(n for n, c in request["games"].items() if c > 0):
        n = request["games"][name]
        r = C.play(layers, load_opponent(PRACTICE[name]), n, PRACTICE_SEED[name], record=True)
        per_opp[name] = C.summary(r)
        rec = r["rec"]
        for i in range(n):
            moves = []
            for t in range(C.MAX_PLY):
                if not rec["valid"][i, t]:
                    break
                e = {"ply": t, "by": "net" if rec["net_moved"][i, t] else "opponent", "move": int(rec["move"][i, t])}
                if rec["net_moved"][i, t]:
                    e["top5"] = [[int(m), round(float(p), 4)] for m, p in zip(rec["top5_i"][i, t], rec["top5_p"][i, t])]
                moves.append(e)
            games.append({"game": gi, "opponent": name, "net_color": "black" if r["net_black"][i] else "white",
                          "result": int(r["result"][i]), "margin": float(r["margin"][i]),
                          "end": C.END_NAMES[int(r["end"][i])], "stones_captured_by_net": int(r["captured"][i]),
                          "stones_lost_by_net": int(r["lost"][i]), "moves": moves})
            gi += 1
        b = np.where(rec["valid"][..., None], rec["board"], 0).astype(np.int8)
        boards.append(b)
        valid.append(rec["valid"].astype(bool))
    files = {"summary.json": json.dumps(per_opp, indent=1).encode(), "README.txt": GAMES_README.encode(),
             "games.json": json.dumps(games).encode(), "boards.npy": _npy_bytes(np.concatenate(boards)),
             "valid.npy": _npy_bytes(np.concatenate(valid))}
    payload = {"ok": True, "pass": k, "kind": "games", "per_opponent": per_opp, "files": f"/work/pass_{k}/",
               "readme": GAMES_README}
    return _wrap("games", k, budget, payload, True), files, True


def games_pass(net, request, out_dir, k=None, budget=None, backend="local", arch=ARCH):
    """Play the requested games, write the pass files into out_dir (only if the request is valid), return the text
    the original harness sent back. `request` is the parsed JSON of the ```games block (a str is parsed here)."""
    k = k if k is not None else _infer_k(out_dir)
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except json.JSONDecodeError as e:
            return _wrap("games", k, budget, {"ok": False, "kind": "games", "error": f"invalid JSON: {e}"}, False)
    if backend == "modal":
        text, files, ok = _remote("games_pass_core", written_net(net, arch), request, k, budget, arch)
    else:
        text, files, ok = games_pass_core(net, request, k, budget, arch)
    if ok:
        _write_files(out_dir, files)
    return text


# ----------------------------------------------------------------------------------------------- positions pass
def _parse_position(p, i):
    from . import core as C
    if not isinstance(p, dict):
        raise ValueError(f"position {i}: must be an object with \"board\" and \"to_move\", or \"moves\"")
    if "moves" in p:
        if set(p) - {"moves"}:
            raise ValueError(f"position {i}: a \"moves\" position takes no other keys")
        if not isinstance(p["moves"], list):
            raise ValueError(f"position {i}: \"moves\" must be a list")
        try:
            board, white, legal = C.replay_moves(p["moves"])
        except ValueError as e:
            raise ValueError(f"position {i}: {e}") from None
        return board, white, legal
    if "board" not in p:
        raise ValueError(f"position {i}: needs \"board\" (with \"to_move\") or \"moves\"")
    if set(p) - {"board", "to_move"}:
        raise ValueError(f"position {i}: unknown keys {sorted(set(p) - {'board', 'to_move'})}")
    b = p["board"]
    if not (isinstance(b, list) and len(b) == 9 and all(isinstance(r, str) and len(r) == 9 for r in b)):
        raise ValueError(f"position {i}: \"board\" must be 9 strings of 9 characters")
    if any(ch not in "XO." for r in b for ch in r):
        raise ValueError(f"position {i}: the board may only contain X, O and .")
    tm = p.get("to_move")
    if not isinstance(tm, str) or tm.lower() not in ("black", "white"):
        raise ValueError(f"position {i}: \"to_move\" must be \"black\" or \"white\"")
    board = np.array([1 if ch == "X" else -1 if ch == "O" else 0 for r in b for ch in r], np.int8)
    legal = np.append(board == 0, True)
    return board, tm.lower() == "white", legal


def _name(m):
    return "pass" if m == 81 else f"({m // 9},{m % 9})"


def positions_pass_core(net, request, k, budget=None, arch=ARCH):
    from . import core as C
    err = None
    if not isinstance(request, dict) or not isinstance(request.get("positions"), list):
        err = 'the request must be JSON {"positions": [...], "activations": true or false}'
    elif set(request) - {"positions", "activations"}:
        err = f"unknown keys {sorted(set(request) - {'positions', 'activations'})}"
    elif not 1 <= len(request["positions"]) <= 256:
        err = f"between 1 and 256 positions ({len(request['positions'])} given)"
    elif not isinstance(request.get("activations", False), bool):
        err = "\"activations\" must be true or false"
    rows = []
    if err is None:
        try:
            rows = [_parse_position(p, i) for i, p in enumerate(request["positions"])]
            t = written_net(net, arch)
        except Exception as e:                                   # noqa: BLE001
            err = str(e)
    if err is not None:
        return _wrap("positions", k, budget, {"ok": False, "kind": "positions", "error": err}, False), {}, False
    acts_on = bool(request.get("activations", False))
    X = np.stack([C.features_np(b, w) for b, w, _ in rows]).astype(np.float32)
    legal = np.stack([l for _, _, l in rows]).astype(bool)
    layers = C.tensors_to_layers(t)
    h, acts = X, []
    for W, b in layers[:-1]:
        h = np.maximum(h @ W + b, 0).astype(np.float32)
        acts.append(h)
    logits = (h @ layers[-1][0] + layers[-1][1]).astype(np.float32)
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    probs = (e / e.sum(1, keepdims=True)).astype(np.float32)
    top3 = []
    for p in probs:
        o = np.argsort(-p, kind="stable")[:3]
        top3.append([[_name(int(m)), round(float(p[m]), 4)] for m in o])
    files = {"inputs.npy": _npy_bytes(X), "legal.npy": _npy_bytes(legal), "logits.npy": _npy_bytes(logits),
             "probs.npy": _npy_bytes(probs)}
    if acts_on:
        for i, a in enumerate(acts, 1):
            files[f"act_L{i}.npy"] = _npy_bytes(a)
    readme = POS_README_BASE + (POS_README_ACT if acts_on else POS_README_NOACT)
    files["README.txt"] = readme.encode()
    payload = {"ok": True, "pass": k, "kind": "positions", "positions": len(rows), "top3_per_position": top3,
               "files": f"/work/pass_{k}/", "readme": readme}
    return _wrap("positions", k, budget, payload, True), files, True


def positions_pass(net, request, out_dir, k=None, budget=None, arch=ARCH, backend="local"):
    """Run the network on the requested positions (host numpy; no GPU needed), write the files, return the text."""
    k = k if k is not None else _infer_k(out_dir)
    if isinstance(request, str):
        try:
            request = json.loads(request)
        except json.JSONDecodeError as e:
            return _wrap("positions", k, budget, {"ok": False, "kind": "positions", "error": f"invalid JSON: {e}"},
                         False)
    text, files, ok = positions_pass_core(net, request, k, budget, arch)
    if ok:
        _write_files(out_dir, files)
    return text


# ----------------------------------------------------------------------------------------------- evaluation
def evaluate(net, games=2000, backend="local", arch=None):
    """Hidden evaluation: `games` games against each hidden opponent (seed 777 + j in the order of HIDDEN, the
    network Black in the first half, both sides sampled). Format of eval/budget_N/eval_submission.json."""
    if backend == "modal":
        return _remote("evaluate", _as_dict(net), games)
    from . import core as C
    t = C.load_net(net)
    layers = C.tensors_to_layers(t)
    out = {}
    for j, (name, opp) in enumerate(HIDDEN.items()):
        out[name] = C.summary(C.play(layers, load_opponent(opp), games, 777 + j))
    out["main_score"] = float(np.mean([out[k]["score"] for k in HIDDEN if k != "random"]))
    out["arch"] = {"depth": len(layers) - 1, "width": int(layers[0][0].shape[1])}
    return out


def h2h(net_a, net_b, games=2000, seed=31337, backend="local"):
    """net_a vs net_b (dicts, folders, opponent names under wsp/go/opponents, or "random"); net_a Black in the first
    half. Returns the summary from net_a's side."""
    if backend == "modal":
        return _remote("h2h", _as_dict(net_a), _as_dict(net_b), games, seed)
    from . import core as C
    a = _resolve(net_a)
    b = _resolve(net_b)
    return C.summary(C.play(a, b, games, seed))


def analysis_strong(net, games=500, seed=4242):
    """The original's diagnostic: 500 games against the practice opponent "strong" (summary part only)."""
    from . import core as C
    return {"opponent": "strong",
            "summary": C.summary(C.play(_resolve(net), load_opponent(PRACTICE["strong"]), games, seed))}


def _resolve(net):
    if isinstance(net, str) and (net == "random" or os.path.isdir(os.path.join(OPP_DIR, net))):
        return load_opponent(net)
    return _layers(net)


def _as_dict(net):
    if isinstance(net, str) and (net == "random" or os.path.isdir(os.path.join(OPP_DIR, net))):
        return net
    from . import core as C
    return C.load_net(net)
