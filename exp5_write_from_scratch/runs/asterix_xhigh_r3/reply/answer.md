# Hand-written policy network for MinAtar Asterix

## Strategy

The player never moves left or right; those logits are fixed at −30. It stays in column 5, where it starts, and only moves up and down. Because each row holds at most one entity (one entity slot per row), and every entity eventually crosses column 5, the whole game reduces to one question per step: which row should the player be in?

- **Dodging.** Flee the current row when an enemy is heading toward column 5 and is 3 or fewer cells away. Never step into a row where an enemy is at column 5, or one cell away and approaching.
- **Gold.** Move toward rows where gold is heading toward column 5. Nearer rows pull harder. If gold is coming in the player's own row, stay and wait for it.
- **Idle.** With nothing nearby, the player usually stays put and drifts gently toward the middle rows.

## Layer 1 (24 of 32 units used)

Each row has at most one entity and one trail cell. The trail shows which way the entity moves, so a weighted sum over the row can tell an approaching entity from a receding one.

**T_r, units 0–7 (enemy threat in row r).** Weights on the enemy and trail cells in row r. Gold cells get −20, so a gold's trail cannot trigger the unit. The pre-activation levels are:

| Enemy in row r | Pre-activation | Output |
|---|---|---|
| At column 5 | +5 | ≈ +1 |
| Approaching, 1 cell away | +4 | ≈ +1 |
| Approaching, 2 cells away | 0 | 0 |
| Approaching, 3 cells away | −1 | −0.76 |
| Approaching, 4 cells away | −2 | −0.96 |
| Receding, absent, or row holds gold | ≤ −3 | ≈ −1 |

**G_r, units 8–15 (approaching gold in row r).** Built the same way with its own trail weights, with −20 on enemy cells. The output is about +1 when gold is coming toward column 5 or is at it, and −1 otherwise.

**P_y, units 16–23.** One-hot code for the player's row: ±0.995.

## Layer 2 (24 of 32 units used)

Each unit is gated by the player's row. It gets +8·P_y (gold units: +10·P_y) and a matching negative bias. When the player is in another row, the unit saturates at exactly −1, a constant that b3 cancels out.

**U_y, units 0–7 (preference for "up" in row y).** The input to tanh is:

2(T_y − T_{y−1}) − 0.2(T_{y−2}+1) − (G_y+1) − 0.4, plus 0.3 in rows 6–8.

The terms mean, in order: danger here minus danger above, a small look-ahead because of sticky actions, stay for gold in this row, a cost for moving, and a pull toward the centre.

**Dn_y, units 8–15 (preference for "down").** The mirror image of U_y. U_1 and Dn_8 are always −1, because those moves do nothing.

**Gd_y, units 16–23 (gold direction).** Σ_j g_j[(G_{y−j}+1) − (G_{y+j}+1)], with g = 1.5, 1.2, 1.0, 0.8, 0.7, 0.6, 0.5 by row distance. Positive means gold is above, negative means below.

## Output

- logit(up) = 14·U + 8·Gd
- logit(down) = 14·Dn − 8·Gd
- logit(no-op) = 0
- left, right and fire = −30

The danger weight (14) is larger than the gold weight (8), so a dangerous target row always vetoes a gold-seeking move.

```edits
# ---------- Layer 1: enemy-threat units T_r (unit r-1), rows 1..8 ----------
W1[61:120:6, 0] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[64:120:6, 0] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[62:120:6, 0] = -20
W1[121:180:6, 1] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[124:180:6, 1] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[122:180:6, 1] = -20
W1[181:240:6, 2] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[184:240:6, 2] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[182:240:6, 2] = -20
W1[241:300:6, 3] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[244:300:6, 3] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[242:300:6, 3] = -20
W1[301:360:6, 4] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[304:360:6, 4] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[302:360:6, 4] = -20
W1[361:420:6, 5] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[364:420:6, 5] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[362:420:6, 5] = -20
W1[421:480:6, 6] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[424:480:6, 6] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[422:480:6, 6] = -20
W1[481:540:6, 7] = [-4, -1, -1.5, 2, 6, 11, 6, 2, -1.5, -1]
W1[484:540:6, 7] = [2, 3.5, 1, 1, -3, -10, -3, 1, 1, 3.5]
W1[482:540:6, 7] = -20
b1[0:8] = -3

# ---------- Layer 1: approaching-gold units G_r (unit 7+r) ----------
W1[62:120:6, 8] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[64:120:6, 8] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[61:120:6, 8] = -20
W1[122:180:6, 9] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[124:180:6, 9] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[121:180:6, 9] = -20
W1[182:240:6, 10] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[184:240:6, 10] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[181:240:6, 10] = -20
W1[242:300:6, 11] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[244:300:6, 11] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[241:300:6, 11] = -20
W1[302:360:6, 12] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[304:360:6, 12] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[301:360:6, 12] = -20
W1[362:420:6, 13] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[364:420:6, 13] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[361:420:6, 13] = -20
W1[422:480:6, 14] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[424:480:6, 14] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[421:480:6, 14] = -20
W1[482:540:6, 15] = [4, 3, 9.5, 11, 16, 17, 16, 11, 9.5, 5]
W1[484:540:6, 15] = [2, -4, -5, -10, -11, -17, -11, -10, -5, -4]
W1[481:540:6, 15] = -20
b1[8:16] = -3

# ---------- Layer 1: player-row one-hot P_y (unit 15+y) ----------
W1[60:120:6, 16] = 6
W1[120:180:6, 17] = 6
W1[180:240:6, 18] = 6
W1[240:300:6, 19] = 6
W1[300:360:6, 20] = 6
W1[360:420:6, 21] = 6
W1[420:480:6, 22] = 6
W1[480:540:6, 23] = 6
b1[16:24] = -3

# ---------- Layer 2: UP danger units U_y (unit y-1) ----------
W2[17, 1] = 8
W2[1, 1] = 2
W2[0, 1] = -2
W2[9, 1] = -1
W2[18, 2] = 8
W2[2, 2] = 2
W2[1, 2] = -2
W2[0, 2] = -0.2
W2[10, 2] = -1
W2[19, 3] = 8
W2[3, 3] = 2
W2[2, 3] = -2
W2[1, 3] = -0.2
W2[11, 3] = -1
W2[20, 4] = 8
W2[4, 4] = 2
W2[3, 4] = -2
W2[2, 4] = -0.2
W2[12, 4] = -1
W2[21, 5] = 8
W2[5, 5] = 2
W2[4, 5] = -2
W2[3, 5] = -0.2
W2[13, 5] = -1
W2[22, 6] = 8
W2[6, 6] = 2
W2[5, 6] = -2
W2[4, 6] = -0.2
W2[14, 6] = -1
W2[23, 7] = 8
W2[7, 7] = 2
W2[6, 7] = -2
W2[5, 7] = -0.2
W2[15, 7] = -1

# ---------- Layer 2: DOWN danger units Dn_y (unit 7+y) ----------
W2[16, 8] = 8
W2[0, 8] = 2
W2[1, 8] = -2
W2[2, 8] = -0.2
W2[8, 8] = -1
W2[17, 9] = 8
W2[1, 9] = 2
W2[2, 9] = -2
W2[3, 9] = -0.2
W2[9, 9] = -1
W2[18, 10] = 8
W2[2, 10] = 2
W2[3, 10] = -2
W2[4, 10] = -0.2
W2[10, 10] = -1
W2[19, 11] = 8
W2[3, 11] = 2
W2[4, 11] = -2
W2[5, 11] = -0.2
W2[11, 11] = -1
W2[20, 12] = 8
W2[4, 12] = 2
W2[5, 12] = -2
W2[6, 12] = -0.2
W2[12, 12] = -1
W2[21, 13] = 8
W2[5, 13] = 2
W2[6, 13] = -2
W2[7, 13] = -0.2
W2[13, 13] = -1
W2[22, 14] = 8
W2[6, 14] = 2
W2[7, 14] = -2
W2[14, 14] = -1
b2[0:16] = [-20, -9.36, -9.56, -9.56, -9.56, -9.26, -9.26, -9.26, -9.26, -9.26, -9.26, -9.56, -9.56, -9.56, -9.36, -20]

# ---------- Layer 2: gold-direction units Gd_y (unit 15+y) ----------
W2[16, 16] = 10
W2[17, 17] = 10
W2[18, 18] = 10
W2[19, 19] = 10
W2[20, 20] = 10
W2[21, 21] = 10
W2[22, 22] = 10
W2[23, 23] = 10
W2[8:16, 16] = [0, -1.5, -1.2, -1.0, -0.8, -0.7, -0.6, -0.5]
W2[8:16, 17] = [1.5, 0, -1.5, -1.2, -1.0, -0.8, -0.7, -0.6]
W2[8:16, 18] = [1.2, 1.5, 0, -1.5, -1.2, -1.0, -0.8, -0.7]
W2[8:16, 19] = [1.0, 1.2, 1.5, 0, -1.5, -1.2, -1.0, -0.8]
W2[8:16, 20] = [0.8, 1.0, 1.2, 1.5, 0, -1.5, -1.2, -1.0]
W2[8:16, 21] = [0.7, 0.8, 1.0, 1.2, 1.5, 0, -1.5, -1.2]
W2[8:16, 22] = [0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 0, -1.5]
W2[8:16, 23] = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 0]
b2[16:24] = [-16.2505, -14.2505, -12.4505, -10.7505, -9.1505, -7.4505, -5.6505, -3.6505]

# ---------- Output layer ----------
W3[0:8, 2] = 14
W3[8:16, 4] = 14
W3[16:24, 2] = 8
W3[16:24, 4] = -8
b3[0:6] = [0, -30, 154, -30, 42, -30]
```