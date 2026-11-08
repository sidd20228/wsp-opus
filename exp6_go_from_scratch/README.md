# Experiment 6: write a 1.5M-weight Go network

The original `message.txt` for each budget, sent unchanged. Opus writes a 243-512x6-82 policy from all-zero weights.

**Harness:**
- **Python:** the original's sandbox prelude, 900 s per run, 96 GB, at most 120 runs; `/work` is kept between runs.
- **Passes:** 0, 10, 25 or 50 games or positions passes (`harness/wsp/go/tools.py`).
- **Effort:** `xhigh`.

**Opponents.** The original's practice and hidden opponents were not released, so they were rebuilt
(`harness/wsp/go/`). Our hidden PPO opponents are weaker than the originals, so Astra's released networks were
re-scored on them for a like-for-like comparison. The head-to-head against the *released* same-size PPO networks is
exact.

**At the snapshot** (`eval/scores.json`; 2,000 games per opponent):

| | vs PPO s0 | vs PPO s2 | vs REINFORCE s0 / s2 | vs random | h2h vs same-size PPO s0 / s1 / s2 |
|---|---|---|---|---|---|
| Astra, budget 0 (on our opponents) | 69.2 | 75.2 | 96.7 / 96.5 | 99.7 | 42.6 / 27.9 / 36.5 |
| Opus, budget 0 | 26.1 | 31.8 | 60.9 / 65.2 | 98.7 | 28.4 / 19.1 / 29.6 |

Budgets 10, 25 and 50 are in progress. `networks/budget_<N>_r<rep>/` holds each finished network, and `runs/` each
conversation. Games-pass per-game records are left out, as in the original release.
