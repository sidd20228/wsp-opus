# Phase 1: Astra (post) vs Opus 5.5

Primary = replicate 1 at the original effort (named before scoring). All runs are listed in scores.json.

## Exp 1: which game (open answer)

| Game | Astra (conf.) | Opus primary (conf.) | Opus high: correct / runs | Opus max: correct / runs |
|---|---|---|---|---|
| breakout | ✓ (99) | breakout (80.0) ✓ | 2 / 2 | 1 / 1 |
| pong | ✓ (96) | pong (75.0) ✓ | 2 / 2 | 1 / 1 |
| pacman | ✓ (96) | pacman (70.0) ✓ | 1 / 1 | 1 / 1 |
| freeway | ✓ (97) | freeway (80.0) ✓ | 1 / 1 | 1 / 1 |

## Exp 2: what the diffusion model draws (open answer)

| Shape | Astra (conf.) | Opus primary (conf.) | Opus high: correct / runs | Opus max: correct / runs | Opus drew filled shapes |
|---|---|---|---|---|---|
| circle | ✓ (84) | circle (55.0) ✓ | 1 / 1 | 1 / 1 | 1 / 2 |
| square | ✓ (77) | square (62.0) ✓ | 1 / 1 | 1 / 1 | 1 / 2 |
| triangle | ✓ (88) | triangle (45.0) ✓ | 1 / 1 | 1 / 1 | 1 / 2 |

## Exp 5: MinAtar networks written from scratch (mean score, 1,000 episodes)

| Game | PPO 100M | PPO 500M | Astra | Opus xhigh (each run) | Opus max (each run) |
|---|---|---|---|---|---|
| breakout | 15.8 | 25.3 | 517.4 | 511.7 | 494.8 |
| seaquest | 5.6 | 88.9 | 53.1 | 49.3 | 41.6 |
| asterix | 25.8 | 55.2 | 53.2 | 49.5 | 64.2 |

## Exp 4: fix the car (success on test episodes; share of weights changed)

| Run | Effort | Statements applied | Weights changed | Success (Kinetix) |
|---|---|---|---|---|
| Astra | high | 51 / 51 | 3.54 % | 1,024 / 1,024 |
| Opus r1 | high | 25 / 25 | 1.33 % | 100.0 % |

## Run health

- runs finished: 23 / 23; failed: 0; all tool-free: True
- list-price equivalent so far: $81.71 (subscription, not billed)
