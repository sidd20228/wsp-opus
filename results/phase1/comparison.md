# Phase 1: Astra (post) vs Opus 5.5

Primary = replicate 1 at the original effort (named before scoring). All runs are listed in scores.json.

## Exp 1: which game (open answer)

| Game | Astra (conf.) | Opus primary (conf.) | Opus high: correct / runs | Opus max: correct / runs |
|---|---|---|---|---|
| breakout | ✓ (99) | breakout (80.0) ✓ | 4 / 4 | 3 / 3 |
| pong | ✓ (96) | pong (75.0) ✓ | 4 / 4 | 3 / 3 |
| pacman | ✓ (96) | pacman (70.0) ✓ | 4 / 4 | 3 / 3 |
| freeway | ✓ (97) | freeway (80.0) ✓ | 3 / 3 | 3 / 3 |

## Exp 2: what the diffusion model draws (open answer)

| Shape | Astra (conf.) | Opus primary (conf.) | Opus high: correct / runs | Opus max: correct / runs | Opus drew filled shapes |
|---|---|---|---|---|---|
| circle | ✓ (84) | circle (55.0) ✓ | 3 / 3 | 3 / 3 | 1 / 6 |
| square | ✓ (77) | square (62.0) ✓ | 3 / 3 | 3 / 3 | 4 / 6 |
| triangle | ✓ (88) | triangle (45.0) ✓ | 3 / 3 | 3 / 3 | 5 / 6 |

## Exp 5: MinAtar networks written from scratch (mean score, 1,000 episodes)

| Game | PPO 100M | PPO 500M | Astra | Opus xhigh (each run) | Opus max (each run) |
|---|---|---|---|---|---|
| breakout | 15.8 | 25.3 | 517.4 | 511.7, 466.9, 378.0 | 494.8, 503.2, 55.9 |
| seaquest | 5.6 | 88.9 | 53.1 | 49.3, 10.4, 11.9 | 41.6, 12.4, 18.1 |
| asterix | 25.8 | 55.2 | 53.2 | 49.5, 74.0, 52.7 | 64.2, 55.3, 32.3 |

## Exp 4: fix the car (success on test episodes; share of weights changed)

| Run | Effort | Statements applied | Weights changed | Success (Kinetix) |
|---|---|---|---|---|
| Astra | high | 51 / 51 | 3.54 % | 1,024 / 1,024 |
| Opus r1 | high | 25 / 25 | 1.33 % | 100.0 % |
| Opus r2 | high | 58 / 58 | 3.57 % | 100.0 % |
| Opus r3 | high | 25 / 25 | 1.33 % | 100.0 % |
| Opus r1 | max | 40 / 40 | 2.45 % | 100.0 % |
| Opus r2 | max | 64 / 64 | 2.15 % | 100.0 % |

## Run health

- runs finished: 68 / 68; failed: 0; all tool-free: True
- list-price equivalent so far: $253.69 (subscription, not billed)
