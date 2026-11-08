# Experiment 1: which game was this network trained on?

The original prompts (`lm_understand-logs/exp1_which_game/runs/<game>/prompt.txt`), sent unchanged. Opus sees one
PPO policy MLP printed in full and must name the game (open answer). Effort `high` (as the original), plus `max` as an
ablation; up to 5 replicates each.

| Game | Astra (conf.) | Opus `high`, primary (conf.) | Opus `max`, r1 (conf.) |
|---|---|---|---|
| Breakout | ✓ 99 | ✓ 80 | ✓ 88 |
| Pong | ✓ 96 | ✓ 75 | ✓ 88 |
| Pac-Man | ✓ 96 | ✓ 70 | ✓ 80 |
| Freeway | ✓ 97 | ✓ 80 | ✓ 85 |

- **Runs:** `runs/<game>_<effort>_r<rep>/`.
- **Scores:** `eval/scores.json`. The answer line is mapped to a game by keyword (`harness/wsp/score_simple.py`). The
  same mapping gives Astra's four answers as scored in the original.
