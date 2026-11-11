**Design**

The policy keeps the paddle under the ball's next column: target T = bx + d, where d is the ball's horizontal direction. It moves right if T > paddle, left if T < paddle, and otherwise does nothing.

- **Normal case.** Usually d = bx − tx, where tx is the trail's column. So S = T − p = 2·bx − tx − p, which is linear in the one-hot inputs.
- **Walls.** When the ball is clamped at a wall, the trail sits in the same column as the ball. Using ball column weights f = [1,2,4,…,16,17] instead of 2c makes S correct there too, including the start position.
- **Paddle edge hit (the special case).** Right after a paddle hit, the ball and the trail are both in row 8. The ball reverses direction if the paddle is under the ball, so the trail points the wrong way. Here the correct rule is sign(bx + tx − 2p), with small wall corrections. It is computed by separate units that are switched on only when both ball and trail are in row 8.
- **Robustness.** This tracker keeps the paddle at most 1 column behind the ball. Sticky actions can only add lag when the action changes, which happens at reversals. Wall rebounds then recover that lag before the next catch.

**Layer 1 (5 units, the rest zero):**
- u0 = tanh(6(S − 0.5)): "S ≥ 1", move right.
- u1 = tanh(6(−S − 0.5)): "S ≤ −1", move left.
- u2: "bx + tx − 2p ≥ 1" (right).
- u3: "bx + tx − 2p ≤ −1" (left).
  - u2 and u3 are switched off by a bias of −240, plus 120 for each of the ball and the trail being in row 8. They only fire when both are in row 8.
- u4 = G: ball and trail both in row 8.

**Layer 2:**
- RIGHT = tanh(4·u0 + 8·u2 − 4·G + 4): this equals u0 normally and u2 when G is on.
- LEFT = tanh(4·u1 + 8·u3 − 4·G + 4), built the same way.

**Output:** logits are right = 14·RIGHT, left = 14·LEFT and no-op = 0; up, down and fire get −10.

```edits
# ---- unit 0: R_n ; unit 1: L_n.  S = f(ball col) - trail col - paddle col
W1[3:600:60, 0] = 6
W1[9:600:60, 0] = 12
W1[15:600:60, 0] = 24
W1[21:600:60, 0] = 36
W1[27:600:60, 0] = 48
W1[33:600:60, 0] = 60
W1[39:600:60, 0] = 72
W1[45:600:60, 0] = 84
W1[51:600:60, 0] = 96
W1[57:600:60, 0] = 102
W1[3:600:60, 1] = -6
W1[9:600:60, 1] = -12
W1[15:600:60, 1] = -24
W1[21:600:60, 1] = -36
W1[27:600:60, 1] = -48
W1[33:600:60, 1] = -60
W1[39:600:60, 1] = -72
W1[45:600:60, 1] = -84
W1[51:600:60, 1] = -96
W1[57:600:60, 1] = -102
# trail (all rows)
W1[10:600:60, 0] = -6
W1[16:600:60, 0] = -12
W1[22:600:60, 0] = -18
W1[28:600:60, 0] = -24
W1[34:600:60, 0] = -30
W1[40:600:60, 0] = -36
W1[46:600:60, 0] = -42
W1[52:600:60, 0] = -48
W1[58:600:60, 0] = -54
W1[10:600:60, 1] = 6
W1[16:600:60, 1] = 12
W1[22:600:60, 1] = 18
W1[28:600:60, 1] = 24
W1[34:600:60, 1] = 30
W1[40:600:60, 1] = 36
W1[46:600:60, 1] = 42
W1[52:600:60, 1] = 48
W1[58:600:60, 1] = 54
# paddle (row 9)
W1[540:600:6, 0] = [0, -6, -12, -18, -24, -30, -36, -42, -48, -54]
W1[540:600:6, 1] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
b1[0] = -3
b1[1] = -3
# ---- units 2,3: after paddle hit (ball & trail in row 8), sign(bx+tx-2p), gated
W1[483:543:6, 2] = [126, 126, 132, 138, 144, 150, 156, 162, 168, 168]
W1[484:544:6, 2] = [120, 126, 132, 138, 144, 150, 156, 162, 168, 174]
W1[540:600:6, 2] = [0, -12, -24, -36, -48, -60, -72, -84, -96, -108]
b1[2] = -243
W1[483:543:6, 3] = [114, 114, 108, 102, 96, 90, 84, 78, 72, 72]
W1[484:544:6, 3] = [120, 114, 108, 102, 96, 90, 84, 78, 72, 66]
W1[540:600:6, 3] = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108]
b1[3] = -243
# ---- unit 4: G = ball and trail both in row 8
W1[483:543:6, 4] = 6
W1[484:544:6, 4] = 6
b1[4] = -9
# ---- layer 2: RIGHT (unit 0), LEFT (unit 1)
W2[0, 0] = 4
W2[2, 0] = 8
W2[4, 0] = -4
b2[0] = 4
W2[1, 1] = 4
W2[3, 1] = 8
W2[4, 1] = -4
b2[1] = 4
# ---- outputs: 0 noop, 1 left, 3 right
W3[0, 3] = 14
W3[1, 1] = 14
b3[2] = -10
b3[4] = -10
b3[5] = -10
```