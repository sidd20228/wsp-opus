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

**All four budgets are done** (`eval/scores.json`; 2,000 games per opponent):

| | vs PPO s0 | vs PPO s2 | vs REINFORCE s0 / s2 | vs random | h2h vs same-size PPO s0 / s1 / s2 |
|---|---|---|---|---|---|
| Astra, budget 0 (on our opponents) | 69.2 | 75.2 | 96.7 / 96.5 | 99.7 | 42.6 / 27.9 / 36.5 |
| Opus, budget 0 | 26.1 | 31.8 | 60.9 / 65.2 | 98.7 | 28.4 / 19.1 / 29.6 |
| Astra, budget 10 (on our opponents) | 68.5 | 88.6 | 91.0 / 88.4 | 99.8 | — |
| Opus, budget 10 | 37.7 | 41.9 | 83.7 / 88.7 | 99.6 | 40.8 / 31.9 / 47.0 |
| Astra, budget 25 (on our opponents) | 64.4 | 82.0 | 85.9 / 88.1 | 98.5 | — |
| Opus, budget 25 | 54.7 | 57.3 | 84.2 / 85.7 | 99.7 | 44.2 / 35.2 / 48.0 |
| Astra, budget 50 (on our opponents) | 90.9 | 91.1 | 95.7 / 97.9 | 99.9 | 74.3 / 69.3 / 81.3 |
| **Opus, budget 50** | **95.2** | **95.6** | **99.7 / 99.7** | 99.9 | **79.6 / 77.0 / 83.2** |

Opus's budget-50 conversation used 117 of 120 Python runs and all 50 passes, over 168 turns. A pattern scan of its code
found no training, fitting, search or running of the network outside a pass. Budgets 10 and 25 used 59 and 81 runs and all their passes; the same scan is clean for every budget. `networks/budget_<N>_r<rep>/` holds each finished network, and `runs/` each
conversation. Games-pass per-game records are left out, as in the original release.
