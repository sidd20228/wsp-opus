# wsp/go: Experiment 6 environment (9x9 Go), opponents, pass tools, hidden evaluation

## Files

| file | what |
|---|---|
| `core.py` | pgx 2.6.0 `go_9x9` rules, 3-plane input, batched play loop (`play`, `play_batch`: vmap over games, scan over 162 plies), `summary` in the original's format, `replay_moves` for "moves" positions |
| `tools.py` | the functions the driver calls: `games_pass`, `positions_pass`, `pass_ok`, `evaluate`, `h2h`, `analysis_strong` |
| `train.py` | PPO / REINFORCE self-play trainers for the 243-32-32-82 opponents (JAX, vmapped over seeds) |
| `modal_app.py` | Modal app `wsp-go` (ephemeral / detached only, nothing deployed), volume `wsp-go`: `bench`, `train_job`, `remote_call` |
| `export_opponents.py` | copies the trained opponents from the volume into `opponents/` (and `wsp-go:opponents/`) |
| `opponents/<name>/` | the six opponents (`L1.W.npy`, `L1.b.npy`, `L2.W.npy`, `L2.b.npy`, `out.W.npy`, `out.b.npy`, `arch.json`, `info.json`) |
| `phase0.py` | Phase 0 check 0.5 + Astra's networks on our opponents -> `results/phase0/go_check.json` |

## How the driver calls it

```python
from wsp.go import tools as G
# net: dict {name: np.ndarray} with the tensors saved so far (missing ones = zeros), or a folder of <name>.npy
#      files (a copy of the sandbox's /work/net, or /work itself).
text = G.games_pass(net, request, out_dir=".../work/pass_7", k=7, budget=50)       # request = parsed ```games JSON
text = G.positions_pass(net, request, out_dir=".../work/pass_8", k=8, budget=50)   # request = parsed ```positions JSON
if G.pass_ok(text): passes_used += 1        # a failed request writes no files and is not counted
# send `text` back to the model verbatim; copy out_dir into the sandbox's /work/pass_<k>/
res = G.evaluate(final_net)                  # eval_submission.json format (2,000 games x 5 opponents)
r = G.h2h(final_net, ".../networks/ppo_same_size_lr3e-5_s0", games=2000, seed=31337)
```

* `text` is exactly what the original harness sent: `Games pass 7 of 50 done.\n```\n{json}\n```\nPasses left: 43.`
  (the JSON has `ok`, `pass`, `kind`, `per_opponent` / `top3_per_position`, `files`, `readme`). Without `budget` only
  the JSON line is returned. `request` may also be the raw JSON string of the block (a JSON error is a failed request).
* A failed request returns `Games pass request failed; it was not counted.\n```\n{"ok": false, "kind": "games",
  "error": "..."}\n```\nPasses left: N.` The original logs contain no failed request, so this wording is ours.
* Validation as the prompt: games only "random"/"weak"/"strong", even non-negative integers, 1..128 in total;
  positions 1..256, each `{"board": [9 strings of 9 of X O .], "to_move": "black"|"white"}` or `{"moves": [...]}`
  (ints 0..81 or "pass", every move must be legal under pgx's rules and the game must not be over), `"activations"`
  a boolean (default false).
* Files (as the original): games pass -> `summary.json` (indent 1), `README.txt`, `games.json` (compact; per game:
  game, opponent, net_color, result, margin, end, stones_captured_by_net, stones_lost_by_net, moves[{ply, by, move,
  top5}]), `boards.npy` (games,162,81) int8, `valid.npy` (games,162) bool. Positions pass -> `README.txt`,
  `inputs.npy`, `legal.npy`, `logits.npy`, `probs.npy` and, if asked, `act_L1.npy .. act_L6.npy`.
* Game order in a games pass: opponents in alphabetical order, the network Black in the first half of each block.
  Each practice opponent has a fixed seed (random 101, strong 102, weak 103), so every pass replays the same random
  stream, as the original did (the random opponent's moves in game 0 are identical in budget_10 and budget_50).
* `backend="local"` (default) runs JAX on the calling machine. On the Mac CPU a 128-game pass takes ~2-3 s and a full
  `evaluate` ~15 s, so no GPU is needed for the passes. `backend="modal"` runs the same code on a Modal GPU in an
  ephemeral app (adds ~1 min of container start). Positions passes always run in numpy on the host.

## Rules and checks

pgx 2.6.0, `pgx._src.games.go.Game(size=9, komi=7.5)` (what `pgx.make("go_9x9")` wraps): komi 7.5; area scoring;
simple ko and suicide masked; a move that recreates any earlier whole-board position (zobrist hash over the whole
game) ends the game and its mover loses; the game ends after two consecutive passes or 162 plies. Network input
(row*9+col)*3+plane from the mover's view, 82 logits, illegal moves masked, temperature-1 sampling, no search.
Matmuls in float32 (`jax_default_matmul_precision=highest`, no TF32).

Checks: the positions pass reproduces Astra's `runs/budget_50/work/pass_36` logits/probs/activations to 3e-5; the
head-to-head of Astra's networks vs the released same-size PPO networks matches the published numbers (see
`results/phase0/go_check.json`); the released same-size PPO networks beat random 97.7-98.3 % here (original
97.5-98.5 %).

## Opponents (the original training code is not released; these are our choices)

All are 243 -> 32 -> 32 -> 82 ReLU (`h = relu(x @ W + b)`, files L1, L2, out), trained by self-play on Modal (H100,
fallback A100/L40S) with `train.py`; each iteration = 1,024 complete games of the learner against an opponent drawn
per game from a pool of 32 frozen snapshots of itself (one every 50 iterations; with probability 0.5 its current
weights), learner Black in half, same rules/input/masking/temperature-1 sampling as the evaluation, reward +1/-1
(pgx result), gamma 1, only the learner's moves are training samples.

| name | role | run | iterations | vs random (2,000 games) |
|---|---|---|---|---|
| ppo_s0 | hidden | `ppo_sc` (PPO, separate 243-32-32-1 critic) | 20,000 | 99.3 % |
| ppo_s2 | hidden | same | 20,000 | 98.8 % |
| ppo_s1 | practice "strong" | same | 20,000 | 97.4 % (prompt: 98 %) |
| reinforce_s0 | hidden | `reinforce` | 5,000 | 95.4 % |
| reinforce_s2 | hidden | same | 5,000 | 95.4 % |
| reinforce_s1 | practice "weak" | same | 5,000 | 94.0 % (prompt: 95 %) |

* PPO: actor 243-32-32-82 + separate critic 243-32-32-1; GAE lambda 0.95; clip 0.2; 4 epochs x 4 minibatches
  (by game); Adam 3e-4 linearly annealed to 0, eps 1e-5; value coef 0.5; entropy 0.01; grad-norm clip 0.5;
  orthogonal init (sqrt 2, policy head 0.01). 20,000 iterations as the original README states.
* REINFORCE: same rollouts, one Adam step (1e-3) per iteration on -mean(logp * (R - mean R)) - 0.01 * entropy,
  no critic. The original gives no iteration count for REINFORCE; we took the checkpoint at 5,000 iterations
  because that is where the practice "weak" network beats random ~95 % as the prompt says (it plateaus at 93-95 %
  from 1,500 to 6,000 iterations and reaches 98 % at 20,000).
* How the recipe was chosen (calibration used only random and the three released same-size PPO networks, never
  Astra's networks): the first recipe (shared policy/value trunk, the `ppo` run; kept on the volume) gave PPO nets
  that beat random 98-99 % but won only ~10 % against the released same-size PPO networks, where the original hidden
  PPO nets were at ~50 % (eval_lr_scan.json). Sweeps at 4,000 iterations (lr 1e-3, lambda 1, entropy 0.001, pure
  self-play, training both sides, score-margin / mixed rewards; runs `sw_*`) did not help; a separate critic did
  (~23 % on average at 20,000 iterations, 10-35 % per seed). Our hidden PPO opponents are still weaker relative to
  the same-size PPO than the original's. Results with the first opponent set: results/phase0/go_check_shared_critic_opponents.json.
