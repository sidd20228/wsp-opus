"""9x9 Go play loop (pgx 2.6.0 `go_9x9` rules), batched with JAX (vmap over games, lax.scan over plies).

Rules come from pgx's own Game object (pgx._src.games.go.Game(size=9, komi=7.5)), which is exactly what
pgx.make("go_9x9") wraps: komi 7.5, area scoring (stones + empty regions reached only by one colour), simple ko and
suicide removed from the legal mask, a move that repeats ANY earlier whole-board position (zobrist hash over the whole
game) ends the game and the mover loses, the game ends after two consecutive passes or 162 plies (size*size*2).
We call Game.step / legal_action_mask / is_terminal / rewards directly (the pgx Env wrapper adds only the
17-plane observation and player bookkeeping, which we do not use).

Network input (the prompt's): 243 = 81 points x 3 planes, index (row*9+col)*3+plane; plane 0 = mover's stones,
plane 1 = opponent's stones, plane 2 = 1 everywhere when White is to move. Output: 82 logits (81 = pass).
A network is a list of (W, b): h = relu(x @ W + b) for every layer but the last, logits = h @ W + b.
Illegal moves are masked; moves are sampled from softmax(masked logits) (temperature 1); no search.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from pgx._src.games import go as _pgo

jax.config.update("jax_default_matmul_precision", "highest")   # no TF32 on GPU: float32 logits as on the CPU

SIZE, NPT, NACT, MAX_PLY, KOMI = 9, 81, 82, 162, 7.5
GAME = _pgo.Game(size=SIZE, komi=KOMI)
assert GAME.max_termination_steps == MAX_PLY

END_NAMES = ("two passes", "repetition", "move limit")


# ----------------------------------------------------------------------------------------------- network helpers
def tensors_to_layers(t):
    """dict of named tensors (L1.W .. Ld.W, L1.b .., out.W, out.b) -> list of (W, b) float32 numpy arrays."""
    depth = 0
    while f"L{depth + 1}.W" in t:
        depth += 1
    layers = [(np.asarray(t[f"L{k}.W"], np.float32), np.asarray(t[f"L{k}.b"], np.float32))
              for k in range(1, depth + 1)]
    layers.append((np.asarray(t["out.W"], np.float32), np.asarray(t["out.b"], np.float32)))
    assert layers[0][0].shape[0] == 3 * NPT and layers[-1][0].shape[1] == NACT, [w.shape for w, _ in layers]
    return layers


def load_net(path_or_dict):
    """A folder of <name>.npy files, an .npz file, or a dict -> dict of float32 numpy tensors."""
    import os
    if isinstance(path_or_dict, dict):
        return {k: np.asarray(v, np.float32) for k, v in path_or_dict.items()}
    p = str(path_or_dict)
    if p.endswith(".npz"):
        z = np.load(p)
        return {k: z[k].astype(np.float32) for k in z.files}
    if os.path.isdir(os.path.join(p, "net")) and not os.path.exists(os.path.join(p, "L1.W.npy")):
        p = os.path.join(p, "net")         # a sandbox /work folder
    out = {}
    for f in os.listdir(p):
        if f.endswith(".npy"):
            out[f[:-4]] = np.load(os.path.join(p, f)).astype(np.float32)
    return out


def mlp(layers, x):
    h = x
    for W, b in layers[:-1]:
        h = jax.nn.relu(h @ W + b)
    W, b = layers[-1]
    return h @ W + b


def mlp_acts(layers, x):
    acts, h = [], x
    for W, b in layers[:-1]:
        h = jax.nn.relu(h @ W + b)
        acts.append(h)
    W, b = layers[-1]
    return h @ W + b, acts


# ----------------------------------------------------------------------------------------------- game helpers
def features(x):
    """GameState -> (243,) float32 input from the mover's view."""
    color = x.color                                   # 0 black to move, 1 white
    my_sign = 1 - 2 * color
    s = jnp.sign(x.board) * my_sign
    planes = jnp.stack([s > 0, s < 0, jnp.broadcast_to(color == 1, s.shape)], axis=-1)
    return planes.reshape(-1).astype(jnp.float32)


def features_np(board_signed, white_to_move):
    """numpy: board (81,) with +1 black / -1 white, to-move flag -> (243,) float32."""
    my = -1 if white_to_move else 1
    s = np.asarray(board_signed) * my
    planes = np.stack([s > 0, s < 0, np.full(NPT, bool(white_to_move))], axis=-1)
    return planes.reshape(-1).astype(np.float32)


def init_batch(n):
    x = jax.vmap(lambda _: GAME.init())(jnp.arange(n))
    legal = jax.vmap(GAME.legal_action_mask)(x)
    return x, legal


def scores(x):
    """(black area, white area) as int32 (vmapped over games)."""
    return jax.vmap(lambda s: _pgo._count_scores(s, SIZE))(x)


def _select_tree(cond, new, old):
    def sel(n, o):
        c = cond.reshape(cond.shape + (1,) * (n.ndim - 1))
        return jnp.where(c, n, o)
    return jax.tree_util.tree_map(sel, new, old)


# ----------------------------------------------------------------------------------------------- play loop
@partial(jax.jit, static_argnames=("n", "opp_random", "net_random", "record"))
def play_batch(net, opp, key, n, net_black, opp_random=False, net_random=False, record=False):
    """Play n games, net vs opp (lists of (W, b); ignored when the matching *_random flag is set: uniform random).
    net_black: (n,) bool, True where the network plays Black. Returns a dict of per-game results (and per-ply records
    when record=True)."""
    x, legal = init_batch(n)
    done = jnp.zeros(n, bool)
    net_passes = jnp.zeros(n, jnp.int32)
    net_pass_early = jnp.zeros(n, bool)
    length = jnp.zeros(n, jnp.int32)

    def body(carry, ply):
        x, legal, done, key, net_passes, net_pass_early, length = carry
        legal_before = legal
        f = jax.vmap(features)(x)
        la = jnp.zeros((n, NACT)) if net_random else mlp(net, f)
        lb = jnp.zeros((n, NACT)) if opp_random else mlp(opp, f)
        net_to_move = (x.color == 0) == net_black
        logits = jnp.where(net_to_move[:, None], la, lb)
        masked = jnp.where(legal, logits, -jnp.inf)
        key, k = jax.random.split(key)
        a = jax.random.categorical(k, masked, axis=-1)
        x2 = jax.vmap(GAME.step)(x, a)
        legal2 = jax.vmap(GAME.legal_action_mask)(x2)
        term2 = jax.vmap(GAME.is_terminal)(x2)
        live = ~done
        x = _select_tree(live, x2, x)
        legal = jnp.where(live[:, None], legal2, legal)
        is_net_pass = live & net_to_move & (a == NPT)
        net_passes = net_passes + is_net_pass
        net_pass_early = net_pass_early | (is_net_pass & (ply < 60))
        length = length + live
        new_done = done | term2
        out = {}
        if record:
            pnet = jax.nn.softmax(jnp.where(legal_before, la, -jnp.inf), axis=-1)
            tp, ti = jax.lax.top_k(pnet, 5)
            out = {"move": jnp.where(live, a, -1).astype(jnp.int16), "valid": live,
                   "board": jnp.sign(x.board).astype(jnp.int8), "net_moved": live & net_to_move,
                   "top5_i": ti.astype(jnp.int16), "top5_p": tp}
        return (x, legal, new_done, key, net_passes, net_pass_early, length), out

    (x, legal, done, key, net_passes, net_pass_early, length), rec = jax.lax.scan(
        body, (x, legal, done, key, net_passes, net_pass_early, length), jnp.arange(MAX_PLY))
    rew = jax.vmap(GAME.rewards)(x)                         # (n, 2): black, white
    net_col = jnp.where(net_black, 0, 1)
    result = jnp.take_along_axis(rew, net_col[:, None], 1)[:, 0]
    sc = scores(x)
    margin_black = sc[:, 0].astype(jnp.float32) - KOMI - sc[:, 1].astype(jnp.float32)
    margin = jnp.where(net_black, margin_black, -margin_black)
    two_pass = x.consecutive_pass_count >= 2
    end = jnp.where(two_pass, 0, jnp.where(x.is_psk, 1, 2))
    cap = x.num_captured                                    # (n, 2): stones captured BY black, BY white
    captured = jnp.take_along_axis(cap, net_col[:, None], 1)[:, 0]
    lost = jnp.take_along_axis(cap, (1 - net_col)[:, None], 1)[:, 0]
    res = {"result": result.astype(jnp.int32), "margin": margin, "end": end, "length": length,
           "captured": captured, "lost": lost, "net_passes": net_passes, "net_pass_early": net_pass_early,
           "net_black": net_black}
    if record:
        res["rec"] = jax.tree_util.tree_map(lambda a: jnp.swapaxes(a, 0, 1), rec)   # (n, 162, ...)
    return res


def to_jax_layers(layers):
    return [(jnp.asarray(W), jnp.asarray(b)) for W, b in layers]


def play(net, opp, n, seed, record=False, chunk=None):
    """net, opp: lists of (W, b) or the string "random". The network is Black in the first half of the games.
    Returns numpy results (concatenated over chunks)."""
    assert n % 2 == 0
    net_black = np.arange(n) < n // 2
    net_random, opp_random = isinstance(net, str), isinstance(opp, str)
    dummy = [(jnp.zeros((3 * NPT, NACT)), jnp.zeros(NACT))]
    jn = dummy if net_random else to_jax_layers(net)
    jo = dummy if opp_random else to_jax_layers(opp)
    key = jax.random.PRNGKey(seed)
    chunk = chunk or n
    outs = []
    for i, s in enumerate(range(0, n, chunk)):
        m = min(chunk, n - s)
        k = key if chunk >= n else jax.random.fold_in(key, i)
        r = play_batch(jn, jo, k, m, jnp.asarray(net_black[s:s + m]), opp_random=opp_random,
                       net_random=net_random, record=record)
        outs.append(jax.tree_util.tree_map(np.asarray, r))
    if len(outs) == 1:
        return outs[0]
    return jax.tree_util.tree_map(lambda *a: np.concatenate(a, 0), *outs)


def summary(r):
    """Per-opponent summary in the original harness's format (float32 for score/margins, float64 for the rest)."""
    res = r["result"]
    nb = r["net_black"]
    g = len(res)
    win = (res > 0)
    f32 = lambda a: float(np.mean(np.asarray(a, np.float32), dtype=np.float32))
    return {
        "games": int(g),
        "score": f32(win),
        "wins": int(win.sum()),
        "losses": int(g - win.sum()),
        "score_as_black": f32(win[nb]) if nb.any() else 0.0,
        "score_as_white": f32(win[~nb]) if (~nb).any() else 0.0,
        "mean_margin": f32(r["margin"]),
        "mean_length": float(np.mean(r["length"].astype(np.float64))),
        "stones_captured_per_game": float(np.mean(r["captured"].astype(np.float64))),
        "stones_lost_per_game": float(np.mean(r["lost"].astype(np.float64))),
        "passes_per_game": float(np.mean(r["net_passes"].astype(np.float64))),
        "games_with_a_pass_before_move_60": int(r["net_pass_early"].sum()),
        "end_two_passes": int((r["end"] == 0).sum()),
        "end_repetition": int((r["end"] == 1).sum()),
        "end_move_limit": int((r["end"] == 2).sum()),
    }


# ----------------------------------------------------------------------------------------------- replay (positions)
def replay_moves(moves):
    """Play a list of moves (ints row*9+col or 'pass') from the empty board with pgx's rules on the host.
    Returns (board (81,) +1 black / -1 white, white_to_move, legal (82,) bool) or raises ValueError."""
    x = GAME.init()
    step = _jit_step()
    for i, m in enumerate(moves):
        if isinstance(m, str):
            if m.strip().lower() != "pass":
                raise ValueError(f"move {i}: {m!r} is not an integer or \"pass\"")
            a = NPT
        elif isinstance(m, (int, np.integer)) and not isinstance(m, bool):
            a = int(m)
            if not 0 <= a <= NPT:
                raise ValueError(f"move {i}: {m} is outside 0..81")
        else:
            raise ValueError(f"move {i}: {m!r} is not an integer or \"pass\"")
        if bool(GAME.is_terminal(x)):
            raise ValueError(f"move {i}: the game is already over")
        legal = np.asarray(_jit_legal()(x))
        if not legal[a]:
            raise ValueError(f"move {i}: {a} is illegal in that position")
        x = step(x, jnp.int32(a))
    if bool(_jit_term()(x)):
        raise ValueError("the game is over after these moves")
    return (np.sign(np.asarray(x.board)).astype(np.int8), bool(int(x.color) == 1), np.asarray(_jit_legal()(x)))


_JIT = {}


def _jit_step():
    if "step" not in _JIT:
        _JIT["step"] = jax.jit(GAME.step)
    return _JIT["step"]


def _jit_legal():
    if "legal" not in _JIT:
        _JIT["legal"] = jax.jit(GAME.legal_action_mask)
    return _JIT["legal"]


def _jit_term():
    if "term" not in _JIT:
        _JIT["term"] = jax.jit(GAME.is_terminal)
    return _JIT["term"]
