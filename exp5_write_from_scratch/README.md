# Experiment 5: write MinAtar networks from scratch

The original prompts, sent unchanged. All weights start at zero, and Opus writes them in one reply with an ```edits
block. Each network (600-32-32-6) is scored over 1,000 episodes, with sticky actions 0.1 and actions sampled from the
softmax (`harness/wsp/minatar_eval.py`).

| Game | PPO 100M | PPO 500M | Astra | Opus `xhigh`, primary | Opus `max`, r1 |
|---|---|---|---|---|---|
| Breakout | 15.8 | 25.3 | 517.4 | 511.7 | 494.8 |
| Seaquest | 5.6 | 88.9 | 53.1 | 49.3 | 41.6 |
| Asterix | 25.8 | 55.2 | 53.2 | 49.5 | 64.2 |

- **Phase 0:** Astra's released networks score within 1.3 standard errors of the published means with this evaluator
  (`results/phase0/minatar_check.json`).
- **Files:** `networks/` holds every network Opus wrote, and `eval/scores.json` the scores with standard errors.
