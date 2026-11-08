**Design**

**Strategy**
- Park the submarine at the left wall, with its front at column 1 and facing right.
- Every enemy that is in or comes into its row will then be in front of it, whether approaching or receding. Divers in the row must cross column 1, so they are collected automatically.
- After that, the policy only moves vertically.
- Its default action is fire. Firing at nothing costs nothing.
- It moves toward rows with targets and stays while a target is in its row.
- It dodges threats near column 1 and never enters a row that has danger near column 1.
- When oxygen drops below 40 and it has at least one diver, it surfaces. Row 1 to row 0 is blocked at all other times.

**Layer 1 (28 units)**
- **T_r** (rows 1–8): target value of row r, from columns 2–9.
  - Trail weights proportional to (x − 9), plus matching enemy/diver weights, make each object score by its direction of motion.
  - Approaching enemy ≈ +1, receding enemy ≈ +0.76, approaching diver ≈ 0, receding diver or empty row ≈ −1.
- **Dn_r** (rows 1–8): danger near the sub's column in row r. This is anything in the danger channel at columns 0–2, or a bullet (danger without reward) at columns 3–5.
- **S_y** (rows 0–8): the submarine is in row y.
- **SM**: surface mode, meaning the diver gauge cell (9,8) is on and the oxygen cell (9,1) is off.
- **H_left**: the submarine is not touching column 0, so it should move left.
- **H_right**: the submarine is at columns 0–1 but facing left. Facing is read from the column-parity of trail and reward cells, where every object–trail pair cancels and only the submarine's back cell remains. This unit means it should press right.

**Layer 2 (18 units)**
- **U_y / D_y**: up and down from row y. Each is gated by S_y (weight 60), so only the submarine's own row is active.
  - It adds distance-weighted attraction to targets above or below.
  - It subtracts a "stay" term when its own row has a target.
  - It adds an escape term when its own row is dangerous.
  - It subtracts a block when the destination row is dangerous.
  - Surface mode pushes up and suppresses down. U_1 only fires in surface mode.
- **D0**: dive from row 0, only once positioned and facing right with row 1 safe.
- **LEFT / RIGHT**: copies of the H units.

**Output**
- The up and down logits are 7 × the sum of their units + 50, giving about +8 when active and about −6 when inactive.
- The fire logit is a constant 3; no-op is −5.

```edits
# ---------- Layer 1: T_r (units 0..7), row target value ----------
W1[70:120:6, 0] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[74:120:6, 0] = [14,12,10,8,6,4,2,0]
W1[73:120:6, 0] = 5
W1[130:180:6, 1] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[134:180:6, 1] = [14,12,10,8,6,4,2,0]
W1[133:180:6, 1] = 5
W1[190:240:6, 2] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[194:240:6, 2] = [14,12,10,8,6,4,2,0]
W1[193:240:6, 2] = 5
W1[250:300:6, 3] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[254:300:6, 3] = [14,12,10,8,6,4,2,0]
W1[253:300:6, 3] = 5
W1[310:360:6, 4] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[314:360:6, 4] = [14,12,10,8,6,4,2,0]
W1[313:360:6, 4] = 5
W1[370:420:6, 5] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[374:420:6, 5] = [14,12,10,8,6,4,2,0]
W1[373:420:6, 5] = 5
W1[430:480:6, 6] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[434:480:6, 6] = [14,12,10,8,6,4,2,0]
W1[433:480:6, 6] = 5
W1[490:540:6, 7] = [-16,-14,-12,-10,-8,-6,-4,-2,0]
W1[494:540:6, 7] = [14,12,10,8,6,4,2,0]
W1[493:540:6, 7] = 5
b1[0:8] = -2
# ---------- Layer 1: Dn_r (units 8..15), danger near column 1 ----------
W1[61:92:6, 8] = [20,20,20,6,6,4]
W1[80:93:6, 8] = [-6,-6,-4]
W1[121:152:6, 9] = [20,20,20,6,6,4]
W1[140:153:6, 9] = [-6,-6,-4]
W1[181:212:6, 10] = [20,20,20,6,6,4]
W1[200:213:6, 10] = [-6,-6,-4]
W1[241:272:6, 11] = [20,20,20,6,6,4]
W1[260:273:6, 11] = [-6,-6,-4]
W1[301:332:6, 12] = [20,20,20,6,6,4]
W1[320:333:6, 12] = [-6,-6,-4]
W1[361:392:6, 13] = [20,20,20,6,6,4]
W1[380:393:6, 13] = [-6,-6,-4]
W1[421:452:6, 14] = [20,20,20,6,6,4]
W1[440:453:6, 14] = [-6,-6,-4]
W1[481:512:6, 15] = [20,20,20,6,6,4]
W1[500:513:6, 15] = [-6,-6,-4]
b1[8:16] = -3
# ---------- Layer 1: S_y (units 16..24), sub row ----------
W1[0:60:6, 16] = 5
W1[60:120:6, 17] = 5
W1[120:180:6, 18] = 5
W1[180:240:6, 19] = 5
W1[240:300:6, 20] = 5
W1[300:360:6, 21] = 5
W1[360:420:6, 22] = 5
W1[420:480:6, 23] = 5
W1[480:540:6, 24] = 5
b1[16:25] = -5
# ---------- Layer 1: surface mode (25) ----------
W1[593, 25] = 6
W1[551, 25] = -6
b1[25] = -3
# ---------- Layer 1: H_left (26) ----------
W1[0:600:60, 26] = -6
b1[26] = 3
# ---------- Layer 1: H_right (27), at col 0 and facing left (parity) ----------
W1[0:600:60, 27] = 6
W1[4:600:12, 27] = -3
W1[10:600:12, 27] = 3
W1[2:600:12, 27] = -3
W1[8:600:12, 27] = 3
W1[2:600:60, 27] = -1.5
W1[56:600:60, 27] = 1.5
b1[27] = -6
# ---------- Layer 2: U_y (units 0..7) ----------
W2[0:8, 0] = [-12.22, -3.05, -2.65, -2.24, -1.93, -1.63, -1.43, -1.22]
W2[17, 0] = 60
W2[8, 0] = 16
W2[25, 0] = 40
b2[0] = -70.03
W2[0:8, 1] = [3.05, -12.22, -3.05, -2.65, -2.24, -1.93, -1.63, -1.43]
W2[18, 1] = 60
W2[9, 1] = 16
W2[8, 1] = -30
W2[25, 1] = 20
b2[1] = -75.91
W2[0:8, 2] = [2.65, 3.05, -12.22, -3.05, -2.65, -2.24, -1.93, -1.63]
W2[19, 2] = 60
W2[10, 2] = 16
W2[9, 2] = -30
W2[25, 2] = 20
b2[2] = -71.98
W2[0:8, 3] = [2.24, 2.65, 3.05, -12.22, -3.05, -2.65, -2.24, -1.93]
W2[20, 3] = 60
W2[11, 3] = 16
W2[10, 3] = -30
W2[25, 3] = 20
b2[3] = -68.25
W2[0:8, 4] = [1.93, 2.24, 2.65, 3.05, -12.22, -3.05, -2.65, -2.24]
W2[21, 4] = 60
W2[12, 4] = 16
W2[11, 4] = -30
W2[25, 4] = 20
b2[4] = -64.52
W2[0:8, 5] = [1.63, 1.93, 2.24, 2.65, 3.05, -12.22, -3.05, -2.65]
W2[22, 5] = 60
W2[13, 5] = 16
W2[12, 5] = -30
W2[25, 5] = 20
b2[5] = -60.79
W2[0:8, 6] = [1.43, 1.63, 1.93, 2.24, 2.65, 3.05, -12.22, -3.05]
W2[23, 6] = 60
W2[14, 6] = 16
W2[13, 6] = -30
W2[25, 6] = 20
b2[6] = -56.86
W2[0:8, 7] = [1.22, 1.43, 1.63, 1.93, 2.24, 2.65, 3.05, -12.22]
W2[24, 7] = 60
W2[15, 7] = 16
W2[14, 7] = -30
W2[25, 7] = 20
b2[7] = -52.73
# ---------- Layer 2: D_y (units 8..14) ----------
W2[0:8, 8] = [-12.22, 3.05, 2.65, 2.24, 1.93, 1.63, 1.43, 1.22]
W2[17, 8] = 60
W2[8, 8] = 16
W2[9, 8] = -30
W2[25, 8] = -10
b2[8] = -82.73
W2[0:8, 9] = [-3.05, -12.22, 3.05, 2.65, 2.24, 1.93, 1.63, 1.43]
W2[18, 9] = 60
W2[9, 9] = 16
W2[10, 9] = -30
W2[25, 9] = -10
b2[9] = -86.86
W2[0:8, 10] = [-2.65, -3.05, -12.22, 3.05, 2.65, 2.24, 1.93, 1.63]
W2[19, 10] = 60
W2[10, 10] = 16
W2[11, 10] = -30
W2[25, 10] = -10
b2[10] = -90.79
W2[0:8, 11] = [-2.24, -2.65, -3.05, -12.22, 3.05, 2.65, 2.24, 1.93]
W2[20, 11] = 60
W2[11, 11] = 16
W2[12, 11] = -30
W2[25, 11] = -10
b2[11] = -94.52
W2[0:8, 12] = [-1.93, -2.24, -2.65, -3.05, -12.22, 3.05, 2.65, 2.24]
W2[21, 12] = 60
W2[12, 12] = 16
W2[13, 12] = -30
W2[25, 12] = -10
b2[12] = -98.25
W2[0:8, 13] = [-1.63, -1.93, -2.24, -2.65, -3.05, -12.22, 3.05, 2.65]
W2[22, 13] = 60
W2[13, 13] = 16
W2[14, 13] = -30
W2[25, 13] = -10
b2[13] = -101.98
W2[0:8, 14] = [-1.43, -1.63, -1.93, -2.24, -2.65, -3.05, -12.22, 3.05]
W2[23, 14] = 60
W2[14, 14] = 16
W2[15, 14] = -30
W2[25, 14] = -10
b2[14] = -105.91
# ---------- Layer 2: D0 (15), LEFT (16), RIGHT (17) ----------
W2[16, 15] = 50
W2[26, 15] = -8
W2[27, 15] = -8
W2[8, 15] = -8
b2[15] = -68
W2[26, 16] = 3
W2[27, 17] = 3
# ---------- Output ----------
W3[0:8, 2] = 7
W3[8:16, 4] = 7
W3[16, 1] = 7
W3[17, 3] = 7
b3[0:6] = [-5, 1, 50, 1, 50, 3]
```