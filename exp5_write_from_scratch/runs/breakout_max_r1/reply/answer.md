**Strategy.** The paddle predicts where the ball will cross row 8. It then moves toward the two cells that can catch the ball, and stops at the first of those cells it reaches.

The catch rule from the source: after the paddle moves, it must sit at `ball_x` or at `ball_x + dx`. Walls act as mirrors at -0.5 and 9.5, so the unfolded landing column is `u = x + dx·(rows to go)`. The catch pair is `[L, R] = fold({u, u+dx})`.

Since `L` and `R` depend only on the ball's cell once the direction is known, the folding is built straight into the ball-cell weights. The policy moves right if `p ≤ L-1`, moves left if `p ≥ R+1`, and does nothing otherwise. This rule has two safety properties:
- If a sticky action makes the paddle overshoot by one cell, it lands on the other catch cell.
- At the walls, the paddle position is clamped, so overshooting there is harmless.

Prediction by phase:
- **Descending ball**, or **ball in rows 0–3** in any direction: exact landing point.
- **Ascending ball in rows 4–8**: assume the ball turns at row 4 (it hits the bottom brick row). Its horizontal travel to row 8 is then `y+1`. This lets the paddle pre-position during the climb. If a hole sends the ball higher, the prediction updates as the ball enters rows 0–3.

**Layer 1 (11 units)**
- **u0 `V_desc`**: on when the ball is descending, or when the ball is in rows 0–5 with the trail in rows 0–4. This also covers trails left by brick hits, the ceiling, and the reset state.
- **u1 `V_asc`**: on when the ball is ascending in rows 4–8, including the horizontal trail just after a paddle bounce.
- **u2 `H`**: +1 if the ball moves right, −1 if it moves left. Its column weights are chosen so that vertical trails after a wall bounce, and trails sitting on the ball's own cell, are classified correctly.
- **u3–u10**: one pair per direction (down-right, down-left, up-right, up-left).
  - The "right" unit computes `tanh(8·(L(ball) − p − 0.5))`.
  - The "left" unit computes `tanh(8·(p − R(ball) − 0.5))`.

**Layer 2 (8 units)**: each unit is an AND of the vertical gate, the horizontal gate (with the right sign) and one comparison unit.

**Output**: the four "right" ANDs feed the right logit and the four "left" ANDs feed the left logit, each ±20. The no-op logit is 0.

```edits
# ===== Layer 1: u0 V_desc =====
W1[3:360:6, 0] = 10
W1[363:420:6, 0] = 20
W1[423:480:6, 0] = 30
W1[483:540:6, 0] = 40
W1[304:360:6, 0] = -10
W1[364:420:6, 0] = -20
W1[424:480:6, 0] = -30
W1[484:540:6, 0] = -40
b1[0] = -5
# ===== u1 V_asc =====
W1[3:240:6, 1] = -60
W1[303:360:6, 1] = -10
W1[363:420:6, 1] = -20
W1[423:540:6, 1] = -30
W1[543:600:6, 1] = -60
W1[304:360:6, 1] = 10
W1[364:420:6, 1] = 20
W1[424:480:6, 1] = 30
W1[484:540:6, 1] = 40
b1[1] = -5
# ===== u2 H (ball columns, then trail columns) =====
W1[15:600:60, 2] = 10
W1[21:600:60, 2] = 10
W1[27:600:60, 2] = 20
W1[33:600:60, 2] = 20
W1[39:600:60, 2] = 30
W1[45:600:60, 2] = 30
W1[51:600:60, 2] = 40
W1[57:600:60, 2] = 40
W1[4:600:60, 2] = 5
W1[10:600:60, 2] = -5
W1[16:600:60, 2] = -5
W1[22:600:60, 2] = -15
W1[28:600:60, 2] = -15
W1[34:600:60, 2] = -25
W1[40:600:60, 2] = -25
W1[46:600:60, 2] = -35
W1[52:600:60, 2] = -35
W1[58:600:60, 2] = -45
# ===== u3 L of down-right =====
W1[3:600:6, 3] = [8,9,8,7,6,5,4,3,2,1, 7,8,9,8,7,6,5,4,3,2,
  6,7,8,9,8,7,6,5,4,3, 5,6,7,8,9,8,7,6,5,4, 4,5,6,7,8,9,8,7,6,5,
  3,4,5,6,7,8,9,8,7,6, 2,3,4,5,6,7,8,9,8,7, 1,2,3,4,5,6,7,8,9,8,
  0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9]
# ===== u4 R of down-right =====
W1[3:600:6, 4] = [9,9,9,8,7,6,5,4,3,2, 8,9,9,9,8,7,6,5,4,3,
  7,8,9,9,9,8,7,6,5,4, 6,7,8,9,9,9,8,7,6,5, 5,6,7,8,9,9,9,8,7,6,
  4,5,6,7,8,9,9,9,8,7, 3,4,5,6,7,8,9,9,9,8, 2,3,4,5,6,7,8,9,9,9,
  1,2,3,4,5,6,7,8,9,9, 1,2,3,4,5,6,7,8,9,9]
# ===== u5 L of down-left =====
W1[3:600:6, 5] = [7,6,5,4,3,2,1,0,0,0, 6,5,4,3,2,1,0,0,0,1,
  5,4,3,2,1,0,0,0,1,2, 4,3,2,1,0,0,0,1,2,3, 3,2,1,0,0,0,1,2,3,4,
  2,1,0,0,0,1,2,3,4,5, 1,0,0,0,1,2,3,4,5,6, 0,0,0,1,2,3,4,5,6,7,
  0,0,1,2,3,4,5,6,7,8, 0,0,1,2,3,4,5,6,7,8]
# ===== u6 R of down-left =====
W1[3:600:6, 6] = [8,7,6,5,4,3,2,1,0,1, 7,6,5,4,3,2,1,0,1,2,
  6,5,4,3,2,1,0,1,2,3, 5,4,3,2,1,0,1,2,3,4, 4,3,2,1,0,1,2,3,4,5,
  3,2,1,0,1,2,3,4,5,6, 2,1,0,1,2,3,4,5,6,7, 1,0,1,2,3,4,5,6,7,8,
  0,1,2,3,4,5,6,7,8,9, 0,1,2,3,4,5,6,7,8,9]
# ===== u7..u10 up-right / up-left (rows 4-8 only) =====
W1[243:540:6, 7] = [5,6,7,8,9,8,7,6,5,4, 6,7,8,9,8,7,6,5,4,3,
  7,8,9,8,7,6,5,4,3,2, 8,9,8,7,6,5,4,3,2,1, 9,8,7,6,5,4,3,2,1,0]
W1[243:540:6, 8] = [6,7,8,9,9,9,8,7,6,5, 7,8,9,9,9,8,7,6,5,4,
  8,9,9,9,8,7,6,5,4,3, 9,9,9,8,7,6,5,4,3,2, 9,9,8,7,6,5,4,3,2,1]
W1[243:540:6, 9] = [4,3,2,1,0,0,0,1,2,3, 5,4,3,2,1,0,0,0,1,2,
  6,5,4,3,2,1,0,0,0,1, 7,6,5,4,3,2,1,0,0,0, 8,7,6,5,4,3,2,1,0,0]
W1[243:540:6, 10] = [5,4,3,2,1,0,1,2,3,4, 6,5,4,3,2,1,0,1,2,3,
  7,6,5,4,3,2,1,0,1,2, 8,7,6,5,4,3,2,1,0,1, 9,8,7,6,5,4,3,2,1,0]
# paddle column p -> -p (before scaling)
W1[546, 3:11] = -1
W1[552, 3:11] = -2
W1[558, 3:11] = -3
W1[564, 3:11] = -4
W1[570, 3:11] = -5
W1[576, 3:11] = -6
W1[582, 3:11] = -7
W1[588, 3:11] = -8
W1[594, 3:11] = -9
b1[3:11:2] = -0.5
b1[4:11:2] = 0.5
# right units: 8*(L-p-0.5); left units: 8*(p-R-0.5)
W1[:, 3:11:2] *= 8
b1[3:11:2] *= 8
W1[:, 4:11:2] *= -8
b1[4:11:2] *= -8
# ===== Layer 2: AND(vertical gate, horizontal gate, comparison) =====
W2[0, 0:4] = 5
W2[1, 4:8] = 5
W2[2, 0:8] = [5, 5, -5, -5, 5, 5, -5, -5]
W2[3, 0] = 5
W2[4, 1] = 5
W2[5, 2] = 5
W2[6, 3] = 5
W2[7, 4] = 5
W2[8, 5] = 5
W2[9, 6] = 5
W2[10, 7] = 5
b2[0:8] = -10
# ===== Output =====
W3[0:8:2, 3] = 20
W3[1:8:2, 1] = 20
b3[1] = 60
b3[3] = 60
```