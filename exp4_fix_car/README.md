# Experiment 4: fix a failing network by editing its weights

The original one-shot prompt (`runs/oneshot_repair/reply/prompt.txt` in the original logs), sent unchanged. Opus
replies once with an ```edits block. The edit is applied to the weights as shown in the prompt
(`harness/wsp/exp4.py`), and the edited actor is scored on Kinetix: the `l/h2_car_ramp` level at commit 80ee9c2, 1,024
sampled episodes from seed 1, plus the greedy episode (`harness/wsp/envs/kinetix_eval.py`).

| Model | Statements applied | Weights changed | Success |
|---|---|---|---|
| unedited | | | 0 / 1,024 |
| Astra | 51 / 51 | 6,073 (3.54 %) | 1,024 / 1,024 |
| Opus `high`, primary | 25 / 25 | 1.33 % | 1,024 / 1,024 (greedy: success) |

- **Phase 0:** Astra's released edit, re-applied here, changes the same 6,073 weights, wins on these episodes, and
  follows the original's logged trajectory step for step (`results/phase0/kinetix_check.json`).
- **Files:** `networks/` holds every edited actor, `runs/*/kinetix.json` the full evaluation, and `eval/scores.json`
  the scores.
