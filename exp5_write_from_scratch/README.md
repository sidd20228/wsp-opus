# Experiment 5: write MinAtar networks from scratch

The original prompts, sent unchanged. All weights start at zero, and Opus writes them in one reply with an ```edits
block. Each network (600-32-32-6) is scored over 1,000 episodes, with sticky actions 0.1 and actions sampled from the
softmax (`harness/wsp/minatar_eval.py`).

| Game | PPO 100M | PPO 500M | Astra | Opus `xhigh` r1 (primary) / r2 / r3 | Opus `max` r1 / r2 / r3 |
|---|---|---|---|---|---|
| Breakout | 15.8 | 25.3 | 517.4 | 511.7 / 466.9 / 378.0 (mean 452.2) | 494.8 / 503.2 / 55.9 |
| Seaquest | 5.6 | 88.9 | 53.1 | 49.3 / 10.4 / 11.9 (mean 23.9) | 41.6 / 12.4 / 18.1 |
| Asterix | 25.8 | 55.2 | 53.2 | 49.5 / 74.0 / 52.7 (mean 58.7) | 64.2 / 55.3 / 32.3 |

The primary networks are close to Astra's, but the replicates vary widely, above all on Seaquest. The post reports one
network per game.

- **Phase 0:** Astra's released networks score within 1.3 standard errors of the published means with this evaluator
  (`results/phase0/minatar_check.json`).
- **Files:** `networks/` holds every network Opus wrote, and `eval/scores.json` the scores with standard errors.
