## Design

**Strategy: move the paddle to the cell where the ball will land.** In this game the ball moves one column per step, which is as fast as the paddle can move. A paddle that only chases the ball keeps a lag of one cell, so one sticky or random action makes it miss. The paddle can gain time only because the ball reflects off the side walls. So the policy computes where the ball will land, with the walls taken into account, and waits there.

**Landing prediction.** I "unfold" the walls: picture the field mirrored at each side wall, so the ball's column moves in a straight line. The game clamps the ball at the wall for one step, so the folding rule is: take u mod 20, keep it if it is 0–9, otherwise use 19 − it.
- **Falling ball:** it reaches row 8 after 8−y more moves.
- **Rising ball:** I assume it turns at the next brick row it has not yet passed (row 3, then 2, then 1, otherwise the ceiling). It therefore lands after y+1 moves if it is in rows 4–8, 6 from row 3, 7 from row 2, 10 from row 1 and 9 from row 0.
- **Why this is safe:** if the ball passes through a gap, the landing point moves by 2 columns but the ball also takes 2 more moves, so the slack is kept. The prediction does not need to read the bricks.

**Which cell to catch on.** The ball is caught from two cells: under the ball, which keeps its horizontal direction, or at the ball's next cell, which reverses it. The target P is whichever of the two is nearer the middle of the field (ties go to "under the ball"). This keeps the following trips short. All of this is a fixed table P(ball cell, horizontal direction, vertical direction), which a one-hot input turns into plain linear weights.

**Layer 1 (10 units used):**
- **Units 0–3 are comparators**, one for each direction case: tanh(4·(P_case(ball cell) − paddle_x)). The output is +1 if the paddle should move right, 0 if it is already at the target, and −1 if it should move left.
- **Unit 4 is the horizontal direction (S).** It computes sign(dx + a(x)), where dx is the ball column minus the trail column. A small correction at x=0 and x=9 handles the wall clamp and the first step of an episode, when there is no movement yet.
- **Unit 5 is the vertical direction (V).** It computes sign(dy + b(row)). The correction b decides which way the ball goes after a step with no vertical movement: after a ceiling or brick hit it goes down, after a paddle hit or a hit from above at row 1 it goes up.
- **Units 6–9 add some randomness.** They fire when the ball is at row 8, falling, over a central column. Their job is to stop the ball looping forever without reaching the remaining bricks.

**Layer 2:**
- **Units 0–7** combine three signals: comparator above +0.5 (or below −0.5), the matching S, and the matching V. This gives a "move right" or "move left" signal for the one direction case that is actually happening.
- **Units 8–11** combine units 6–9 with S.

**Output logits:**
- right or left gets 30 when that move is needed;
- no-op gets 15;
- the randomness units add 15 toward the other catching cell, giving a 50/50 choice there that is always still a catch;
- up, down and fire are suppressed at −50.

```edits
# ===== Layer 1 =====
# unit 0: comparator, case (moving right, going down): tanh(4*(P(ball) - paddle_x))
W1[3:600:6, 0] = [8,9,8,7,6,5,5,4,3,2,
  7,8,9,8,7,6,5,5,4,3,
  6,7,8,9,8,7,6,5,5,4,
  5,6,7,8,9,8,7,6,5,5,
  4,5,6,7,8,9,8,7,6,5,
  4,4,5,6,7,8,9,8,7,6,
  3,4,4,5,6,7,8,9,8,7,
  2,3,4,4,5,6,7,8,9,8,
  1,2,3,4,4,5,6,7,8,9,
  0,0,0,0,0,0,0,0,0,0]
W1[540:600:6, 0] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 0] *= 4
# unit 1: comparator, case (moving left, going down)
W1[3:600:6, 1] = [7,6,5,4,4,3,2,1,0,1,
  6,5,4,4,3,2,1,0,1,2,
  5,4,4,3,2,1,0,1,2,3,
  4,4,3,2,1,0,1,2,3,4,
  4,3,2,1,0,1,2,3,4,5,
  3,2,1,0,1,2,3,4,5,5,
  2,1,0,1,2,3,4,5,5,6,
  1,0,1,2,3,4,5,5,6,7,
  0,1,2,3,4,5,5,6,7,8,
  0,0,0,0,0,0,0,0,0,0]
W1[540:600:6, 1] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 1] *= 4
# unit 2: comparator, case (moving right, going up; assumes turn at next brick row)
W1[3:600:6, 2] = [9,8,7,6,5,5,4,3,2,1,
  8,7,6,5,5,4,3,2,1,0,
  7,8,9,8,7,6,5,5,4,3,
  6,7,8,9,8,7,6,5,5,4,
  5,6,7,8,9,8,7,6,5,5,
  6,7,8,9,8,7,6,5,5,4,
  7,8,9,8,7,6,5,5,4,3,
  8,9,8,7,6,5,5,4,3,2,
  9,8,7,6,5,5,4,3,2,1,
  0,0,0,0,0,0,0,0,0,0]
W1[540:600:6, 2] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 2] *= 4
# unit 3: comparator, case (moving left, going up)
W1[3:600:6, 3] = [8,7,6,5,4,4,3,2,1,0,
  9,8,7,6,5,4,4,3,2,1,
  6,5,4,4,3,2,1,0,1,2,
  5,4,4,3,2,1,0,1,2,3,
  4,4,3,2,1,0,1,2,3,4,
  5,4,4,3,2,1,0,1,2,3,
  6,5,4,4,3,2,1,0,1,2,
  7,6,5,4,4,3,2,1,0,1,
  8,7,6,5,4,4,3,2,1,0,
  0,0,0,0,0,0,0,0,0,0]
W1[540:600:6, 3] = [0,-1,-2,-3,-4,-5,-6,-7,-8,-9]
W1[:, 3] *= 4
# unit 4: S = sign(ball_x - trail_x + wall correction)  (+1 = moving right)
W1[3:600:60, 4] = 0.5
W1[9:600:60, 4] = 1
W1[15:600:60, 4] = 2
W1[21:600:60, 4] = 3
W1[27:600:60, 4] = 4
W1[33:600:60, 4] = 5
W1[39:600:60, 4] = 6
W1[45:600:60, 4] = 7
W1[51:600:60, 4] = 8
W1[57:600:60, 4] = 8.5
W1[10:600:60, 4] = -1
W1[16:600:60, 4] = -2
W1[22:600:60, 4] = -3
W1[28:600:60, 4] = -4
W1[34:600:60, 4] = -5
W1[40:600:60, 4] = -6
W1[46:600:60, 4] = -7
W1[52:600:60, 4] = -8
W1[58:600:60, 4] = -9
W1[:, 4] *= 8
# unit 5: V = sign(ball_row - trail_row + row correction)  (+1 = going down)
W1[3:60:6, 5] = 0.5
W1[63:120:6, 5] = 0.5
W1[123:180:6, 5] = 2.5
W1[183:240:6, 5] = 3.5
W1[243:300:6, 5] = 4.5
W1[303:360:6, 5] = 5
W1[363:420:6, 5] = 6
W1[423:480:6, 5] = 7
W1[483:540:6, 5] = 7.5
W1[543:600:6, 5] = 9
W1[64:120:6, 5] = -1
W1[124:180:6, 5] = -2
W1[184:240:6, 5] = -3
W1[244:300:6, 5] = -4
W1[304:360:6, 5] = -5
W1[364:420:6, 5] = -6
W1[424:480:6, 5] = -7
W1[484:540:6, 5] = -8
W1[544:600:6, 5] = -9
W1[:, 5] *= 8
# units 6-9: final-step randomisation detectors (ball row 8, trail row 7)
W1[507:531:6, 6] = 8
W1[442:466:6, 6] = 8
W1[489:507:6, 7] = 8
W1[424:442:6, 7] = 8
W1[519:537:6, 8] = 8
W1[466:484:6, 8] = 8
W1[495:519:6, 9] = 8
W1[442:466:6, 9] = 8
b1[6:10] = -12

# ===== Layer 2 =====
W2[0, 0:12] = [10,0,0,0,-10,0,0,0,0,0,0,0]
W2[1, 0:12] = [0,10,0,0,0,-10,0,0,0,0,0,0]
W2[2, 0:12] = [0,0,10,0,0,0,-10,0,0,0,0,0]
W2[3, 0:12] = [0,0,0,10,0,0,0,-10,0,0,0,0]
W2[4, 0:12] = [10,-10,10,-10,10,-10,10,-10,10,10,-10,-10]
W2[5, 0:12] = [10,10,-10,-10,10,10,-10,-10,0,0,0,0]
W2[6, 8] = 10
W2[7, 9] = 10
W2[8, 10] = 10
W2[9, 11] = 10
b2[0:8] = -25
b2[8:12] = -15

# ===== Output =====
W3[0:4, 3] = 15
W3[4:8, 1] = 15
W3[8, 3] = 7.5
W3[10, 3] = 7.5
W3[9, 1] = 7.5
W3[11, 1] = 7.5
b3[0:6] = [15, 75, -50, 75, -50, -50]
```