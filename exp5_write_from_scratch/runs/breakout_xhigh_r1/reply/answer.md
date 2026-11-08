## Design

**Strategy: catch-window prediction.** The paddle never chases the ball. It goes straight to where the ball can be caught, then waits there.

The ball is caught when, after the paddle moves, the paddle sits in either of two columns:
- the column the ball occupies on row 8, or
- the column the ball is about to move into.

I call this 1–2 cell set the catch window.

**Predicting the window.**
- **Direction.** The ball's direction comes from the trail cell.
- **Walls.** Wall reflections are unfolded: a reflected position is computed as if the walls were not there, then folded back. The fold has period 20 with the end columns doubled.
- **Falling ball.** The window is exact: {fold(u), fold(u±1)}, where u = x ± (8−y).
- **Rising ball.** The network assumes the ball rebounds off the lowest brick row (bounce at row 4). In rows 2–3 it assumes an immediate rebound, and in rows 0–1 a bounce off the top wall.

**Why this is safe.** If the rising-ball guess is wrong, the error is at most 2 cells per extra row the ball climbs. That is always less than the extra descent time it gives, so the paddle can still correct.

**Rule.** Move toward the nearest cell of the window, and no-op once inside it. Stopping at the near edge matters: if a sticky action repeats the arrival move, the overshoot lands in the other window cell and the ball is still caught. Prediction targets stay constant along each flight, so the action rarely changes, which limits harm from sticky actions. In deterministic play the ball is never missed. With every cell reachable over time, the ball cannot loop forever without hitting bricks.

**Layer 1 (12 units).** Ball, trail and paddle are each one-hot, so any lookup table of a cell is a linear weight.

| Unit | Role | Pre-activation (×10) | Fires +1 when |
|---|---|---|---|
| 0 R | horizontal direction | (x + ½·[x=0] − ½·[x=9]) − x_trail | ball moves right; the ± ½ terms handle wall bounces |
| 1 D | vertical direction | (y + ½·[y≤4] − ½·[y=8]) − y_trail | ball moves down; covers brick/top-wall bounces and the paddle bounce |
| 2 R8 | outgoing direction right after a paddle hit | x + x_trail − 2·paddle (with wall terms) | ball leaves right; corrects the trail, which is wrong after a paddle reversal |
| 3 B8 | "just hit the paddle" | ball in row 8 AND trail in row 8 | paddle hit occurred |
| 4–11 | 8 comparators, 2 per direction hypothesis (UR, UL, DR, DL) | 10·(lo(ball) − paddle − ½) or 10·(paddle − hi(ball) − ½) | window is right of paddle, or left of paddle |

In the comparators, lo and hi are precomputed lookup tables for the low and high cell of the catch window.

**Layer 2 (12 units).** These are AND gates: each selects the comparator for the ball's actual direction (UR, UL, DR, DL, plus up-right/up-left just after a paddle hit), separately for "go right" and "go left."

**Output.**
- Right and left logits are each +20 if one of their gates fires, otherwise −20.
- No-op is 0.
- Up, down and fire are −30.

```edits
# ===== Layer 1 =====
# Unit 0: R = horizontal direction (+1 = moving right)
W1[3:600:60, 0] = 5
W1[9:600:60, 0] = 10
W1[15:600:60, 0] = 20
W1[21:600:60, 0] = 30
W1[27:600:60, 0] = 40
W1[33:600:60, 0] = 50
W1[39:600:60, 0] = 60
W1[45:600:60, 0] = 70
W1[51:600:60, 0] = 80
W1[57:600:60, 0] = 85
W1[10:600:60, 0] = -10
W1[16:600:60, 0] = -20
W1[22:600:60, 0] = -30
W1[28:600:60, 0] = -40
W1[34:600:60, 0] = -50
W1[40:600:60, 0] = -60
W1[46:600:60, 0] = -70
W1[52:600:60, 0] = -80
W1[58:600:60, 0] = -90
# Unit 1: D = vertical direction (+1 = moving down)
W1[3:60:6, 1] = 5
W1[63:120:6, 1] = 15
W1[123:180:6, 1] = 25
W1[183:240:6, 1] = 35
W1[243:300:6, 1] = 45
W1[303:360:6, 1] = 50
W1[363:420:6, 1] = 60
W1[423:480:6, 1] = 70
W1[483:540:6, 1] = 75
W1[543:600:6, 1] = 90
W1[64:120:6, 1] = -10
W1[124:180:6, 1] = -20
W1[184:240:6, 1] = -30
W1[244:300:6, 1] = -40
W1[304:360:6, 1] = -50
W1[364:420:6, 1] = -60
W1[424:480:6, 1] = -70
W1[484:540:6, 1] = -80
W1[544:600:6, 1] = -90
# Unit 2: R8 = outgoing horizontal direction just after a paddle hit
W1[3:600:60, 2] = 5
W1[9:600:60, 2] = 10
W1[15:600:60, 2] = 20
W1[21:600:60, 2] = 30
W1[27:600:60, 2] = 40
W1[33:600:60, 2] = 50
W1[39:600:60, 2] = 60
W1[45:600:60, 2] = 70
W1[51:600:60, 2] = 80
W1[57:600:60, 2] = 85
W1[10:600:60, 2] = 10
W1[16:600:60, 2] = 20
W1[22:600:60, 2] = 30
W1[28:600:60, 2] = 40
W1[34:600:60, 2] = 50
W1[40:600:60, 2] = 60
W1[46:600:60, 2] = 70
W1[52:600:60, 2] = 80
W1[58:600:60, 2] = 90
W1[540:600:6, 2] = [0, -20, -40, -60, -80, -100, -120, -140, -160, -180]
# Unit 3: B8 = ball and trail both on row 8 (just bounced off paddle)
W1[483:540:6, 3] = 10
W1[484:540:6, 3] = 10
b1[3] = -15
# Unit 4: window-low(UR) > paddle
W1[3:600:6, 4] = [90,80,70,60,50,40,30,20,10,0,
 80,70,60,50,40,30,20,10,0,0,
 70,80,90,80,70,60,50,40,30,20,
 60,70,80,90,80,70,60,50,40,30,
 50,60,70,80,90,80,70,60,50,40,
 60,70,80,90,80,70,60,50,40,30,
 70,80,90,80,70,60,50,40,30,20,
 80,90,80,70,60,50,40,30,20,10,
 90,80,70,60,50,40,30,20,10,0,
 90,80,70,60,50,40,30,20,10,0]
W1[540:600:6, 4] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
b1[4] = -5
# Unit 5: paddle > window-high(UR)
W1[3:600:6, 5] = [-90,-90,-80,-70,-60,-50,-40,-30,-20,-10,
 -90,-80,-70,-60,-50,-40,-30,-20,-10,0,
 -80,-90,-90,-90,-80,-70,-60,-50,-40,-30,
 -70,-80,-90,-90,-90,-80,-70,-60,-50,-40,
 -60,-70,-80,-90,-90,-90,-80,-70,-60,-50,
 -70,-80,-90,-90,-90,-80,-70,-60,-50,-40,
 -80,-90,-90,-90,-80,-70,-60,-50,-40,-30,
 -90,-90,-90,-80,-70,-60,-50,-40,-30,-20,
 -90,-90,-80,-70,-60,-50,-40,-30,-20,-10,
 -90,-90,-80,-70,-60,-50,-40,-30,-20,-10]
W1[540:600:6, 5] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
b1[5] = -5
# Unit 6: window-low(UL) > paddle
W1[3:600:6, 6] = [80,70,60,50,40,30,20,10,0,0,
 90,80,70,60,50,40,30,20,10,0,
 60,50,40,30,20,10,0,0,0,10,
 50,40,30,20,10,0,0,0,10,20,
 40,30,20,10,0,0,0,10,20,30,
 50,40,30,20,10,0,0,0,10,20,
 60,50,40,30,20,10,0,0,0,10,
 70,60,50,40,30,20,10,0,0,0,
 80,70,60,50,40,30,20,10,0,0,
 80,70,60,50,40,30,20,10,0,0]
W1[540:600:6, 6] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
b1[6] = -5
# Unit 7: paddle > window-high(UL)
W1[3:600:6, 7] = [-90,-80,-70,-60,-50,-40,-30,-20,-10,0,
 -90,-90,-80,-70,-60,-50,-40,-30,-20,-10,
 -70,-60,-50,-40,-30,-20,-10,0,-10,-20,
 -60,-50,-40,-30,-20,-10,0,-10,-20,-30,
 -50,-40,-30,-20,-10,0,-10,-20,-30,-40,
 -60,-50,-40,-30,-20,-10,0,-10,-20,-30,
 -70,-60,-50,-40,-30,-20,-10,0,-10,-20,
 -80,-70,-60,-50,-40,-30,-20,-10,0,-10,
 -90,-80,-70,-60,-50,-40,-30,-20,-10,0,
 -90,-80,-70,-60,-50,-40,-30,-20,-10,0]
W1[540:600:6, 7] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
b1[7] = -5
# Unit 8: window-low(DR) > paddle
W1[3:600:6, 8] = [80,90,80,70,60,50,40,30,20,10,
 70,80,90,80,70,60,50,40,30,20,
 60,70,80,90,80,70,60,50,40,30,
 50,60,70,80,90,80,70,60,50,40,
 40,50,60,70,80,90,80,70,60,50,
 30,40,50,60,70,80,90,80,70,60,
 20,30,40,50,60,70,80,90,80,70,
 10,20,30,40,50,60,70,80,90,80,
 0,10,20,30,40,50,60,70,80,90,
 0,10,20,30,40,50,60,70,80,90]
W1[540:600:6, 8] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
b1[8] = -5
# Unit 9: paddle > window-high(DR)
W1[3:600:6, 9] = [-90,-90,-90,-80,-70,-60,-50,-40,-30,-20,
 -80,-90,-90,-90,-80,-70,-60,-50,-40,-30,
 -70,-80,-90,-90,-90,-80,-70,-60,-50,-40,
 -60,-70,-80,-90,-90,-90,-80,-70,-60,-50,
 -50,-60,-70,-80,-90,-90,-90,-80,-70,-60,
 -40,-50,-60,-70,-80,-90,-90,-90,-80,-70,
 -30,-40,-50,-60,-70,-80,-90,-90,-90,-80,
 -20,-30,-40,-50,-60,-70,-80,-90,-90,-90,
 -10,-20,-30,-40,-50,-60,-70,-80,-90,-90,
 -10,-20,-30,-40,-50,-60,-70,-80,-90,-90]
W1[540:600:6, 9] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
b1[9] = -5
# Unit 10: window-low(DL) > paddle
W1[3:600:6, 10] = [70,60,50,40,30,20,10,0,0,0,
 60,50,40,30,20,10,0,0,0,10,
 50,40,30,20,10,0,0,0,10,20,
 40,30,20,10,0,0,0,10,20,30,
 30,20,10,0,0,0,10,20,30,40,
 20,10,0,0,0,10,20,30,40,50,
 10,0,0,0,10,20,30,40,50,60,
 0,0,0,10,20,30,40,50,60,70,
 0,0,10,20,30,40,50,60,70,80,
 0,0,10,20,30,40,50,60,70,80]
W1[540:600:6, 10] = [0, -10, -20, -30, -40, -50, -60, -70, -80, -90]
b1[10] = -5
# Unit 11: paddle > window-high(DL)
W1[3:600:6, 11] = [-80,-70,-60,-50,-40,-30,-20,-10,0,-10,
 -70,-60,-50,-40,-30,-20,-10,0,-10,-20,
 -60,-50,-40,-30,-20,-10,0,-10,-20,-30,
 -50,-40,-30,-20,-10,0,-10,-20,-30,-40,
 -40,-30,-20,-10,0,-10,-20,-30,-40,-50,
 -30,-20,-10,0,-10,-20,-30,-40,-50,-60,
 -20,-10,0,-10,-20,-30,-40,-50,-60,-70,
 -10,0,-10,-20,-30,-40,-50,-60,-70,-80,
 0,-10,-20,-30,-40,-50,-60,-70,-80,-90,
 0,-10,-20,-30,-40,-50,-60,-70,-80,-90]
W1[540:600:6, 11] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
b1[11] = -5
# ===== Layer 2: AND gates (comparator x active direction) =====
# 0 right_UR
W2[4, 0] = 10
W2[0, 0] = 10
W2[1, 0] = -10
W2[3, 0] = -10
# 1 right_UL
W2[6, 1] = 10
W2[0, 1] = -10
W2[1, 1] = -10
W2[3, 1] = -10
# 2 right_DR
W2[8, 2] = 10
W2[0, 2] = 10
W2[1, 2] = 10
# 3 right_DL
W2[10, 3] = 10
W2[0, 3] = -10
W2[1, 3] = 10
# 4 right_UR just after paddle hit
W2[4, 4] = 10
W2[2, 4] = 10
W2[3, 4] = 10
# 5 right_UL just after paddle hit
W2[6, 5] = 10
W2[2, 5] = -10
W2[3, 5] = 10
# 6 left_UR
W2[5, 6] = 10
W2[0, 6] = 10
W2[1, 6] = -10
W2[3, 6] = -10
# 7 left_UL
W2[7, 7] = 10
W2[0, 7] = -10
W2[1, 7] = -10
W2[3, 7] = -10
# 8 left_DR
W2[9, 8] = 10
W2[0, 8] = 10
W2[1, 8] = 10
# 9 left_DL
W2[11, 9] = 10
W2[0, 9] = -10
W2[1, 9] = 10
# 10 left_UR just after paddle hit
W2[5, 10] = 10
W2[2, 10] = 10
W2[3, 10] = 10
# 11 left_UL just after paddle hit
W2[7, 11] = 10
W2[2, 11] = -10
W2[3, 11] = 10
b2[0:12] = [-30, -30, -20, -20, -20, -20, -30, -30, -20, -20, -20, -20]
# ===== Output =====
W3[0:6, 3] = 20
W3[6:12, 1] = 20
b3[0:6] = [0, 100, -30, 100, -30, -30]
```