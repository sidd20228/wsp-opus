## Design

**Strategy.** The submarine is kept parked in columns 5–6. Facing left means the front is at column 5; facing right means the front is at column 6. A single left or right press only turns it, so it never drifts. This makes orientation readable: the back cell is the cell that is also lit in the trail channel. It also makes "enemy to my left or right" a fixed set of columns.

From that position the policy works as follows:

1. **Choosing a row.** It moves vertically toward the topmost target row (rows 2–8; targets are enemies or divers in the reward channel). It holds its row while any target is in it.
2. **Safe vertical moves.** It only moves up or down when the destination row has no danger in columns 3–8.
3. **Shooting.** When danger in its row is behind it, it turns to face it. Otherwise firing is the default action. Divers arrive on their own, because they always move toward the centre.
4. **Never surfacing empty.** At rows 0–1 it goes down when the row below is clear. "Up" carries a −10 bias, and no feature pushes up from rows 1–2. The one exception is surfacing.
5. **Surfacing.** If oxygen is below 60 (gauge cell (9,2) is off) and at least one diver is aboard (gauge cell (9,8) is on), it goes up with top priority.
6. **Recovery.** If a sticky repeat or noise pushes the submarine off columns 5–6, it steps back.

**Layer-1 units** (each is an AND gate; on gives +3 pre-activation, off gives −3 or less):
- **u0–u6, TurnL (rows 2–8):** sub in row r, facing right (trail at (r,5)), and danger in columns 0–4 of row r.
- **u7–u13, TurnR (rows 2–8):** sub in row r, facing left (trail at (r,6)), and danger in columns 7–9 of row r.
- **u14–u19, Up from row s (s = 3–8):** sub in row s, a target in rows 2..s−1, no target in row s, and no danger at (s−1, columns 3–8).
- **u20–u25, Down from row s (s = 2–7):** sub in row s, a target in rows s+1..8, no target in rows 2..s, and no danger at (s+1, columns 3–8).
- **u26, u27:** sub in row 0 (or row 1) and the next row clear near the centre.
- **u28:** surface condition.
- **u29, u30:** sub is too far right or too far left.

**Layer 2:** ORs each group into one 0/1 feature: tanh(2·Σh + 2·n).

**Output logits:**
- Default is fire at 6.
- Turn adds +12 to left or right.
- Vertical moves score 14.
- Recovery scores 16.
- Surfacing scores 26 on up.

```edits
# ---------- Layer 1: TurnL rows 2..8 (units 0..6) ----------
W1[120:180:6, 0] = 30
W1[154, 0] = 18
W1[121:151:6, 0] = 6
W1[180:240:6, 1] = 30
W1[214, 1] = 18
W1[181:211:6, 1] = 6
W1[240:300:6, 2] = 30
W1[274, 2] = 18
W1[241:271:6, 2] = 6
W1[300:360:6, 3] = 30
W1[334, 3] = 18
W1[301:331:6, 3] = 6
W1[360:420:6, 4] = 30
W1[394, 4] = 18
W1[361:391:6, 4] = 6
W1[420:480:6, 5] = 30
W1[454, 5] = 18
W1[421:451:6, 5] = 6
W1[480:540:6, 6] = 30
W1[514, 6] = 18
W1[481:511:6, 6] = 6
b1[0:7] = -81
# ---------- TurnR rows 2..8 (units 7..13) ----------
W1[120:180:6, 7] = 30
W1[160, 7] = 18
W1[163:181:6, 7] = 6
W1[180:240:6, 8] = 30
W1[220, 8] = 18
W1[223:241:6, 8] = 6
W1[240:300:6, 9] = 30
W1[280, 9] = 18
W1[283:301:6, 9] = 6
W1[300:360:6, 10] = 30
W1[340, 10] = 18
W1[343:361:6, 10] = 6
W1[360:420:6, 11] = 30
W1[400, 11] = 18
W1[403:421:6, 11] = 6
W1[420:480:6, 12] = 30
W1[460, 12] = 18
W1[463:481:6, 12] = 6
W1[480:540:6, 13] = 30
W1[520, 13] = 18
W1[523:541:6, 13] = 6
b1[7:14] = -81
# ---------- Up from row s=3..8 (units 14..19) ----------
W1[180:240:6, 14] = 30
W1[122:182:6, 14] = 6
W1[182:242:6, 14] = -30
W1[139:175:6, 14] = -30
W1[240:300:6, 15] = 30
W1[122:242:6, 15] = 6
W1[242:302:6, 15] = -30
W1[199:235:6, 15] = -30
W1[300:360:6, 16] = 30
W1[122:302:6, 16] = 6
W1[302:362:6, 16] = -30
W1[259:295:6, 16] = -30
W1[360:420:6, 17] = 30
W1[122:362:6, 17] = 6
W1[362:422:6, 17] = -30
W1[319:355:6, 17] = -30
W1[420:480:6, 18] = 30
W1[122:422:6, 18] = 6
W1[422:482:6, 18] = -30
W1[379:415:6, 18] = -30
W1[480:540:6, 19] = 30
W1[122:482:6, 19] = 6
W1[482:542:6, 19] = -30
W1[439:475:6, 19] = -30
b1[14:20] = -63
# ---------- Down from row s=2..7 (units 20..25) ----------
W1[120:180:6, 20] = 30
W1[182:542:6, 20] = 6
W1[122:182:6, 20] = -30
W1[199:235:6, 20] = -30
W1[180:240:6, 21] = 30
W1[242:542:6, 21] = 6
W1[122:242:6, 21] = -30
W1[259:295:6, 21] = -30
W1[240:300:6, 22] = 30
W1[302:542:6, 22] = 6
W1[122:302:6, 22] = -30
W1[319:355:6, 22] = -30
W1[300:360:6, 23] = 30
W1[362:542:6, 23] = 6
W1[122:362:6, 23] = -30
W1[379:415:6, 23] = -30
W1[360:420:6, 24] = 30
W1[422:542:6, 24] = 6
W1[122:422:6, 24] = -30
W1[439:475:6, 24] = -30
W1[420:480:6, 25] = 30
W1[482:542:6, 25] = 6
W1[122:482:6, 25] = -30
W1[499:535:6, 25] = -30
b1[20:26] = -63
# ---------- Dive from rows 0/1 (units 26,27) ----------
W1[0:60:6, 26] = 30
W1[79:115:6, 26] = -30
b1[26] = -57
W1[60:120:6, 27] = 30
W1[139:175:6, 27] = -30
b1[27] = -57
# ---------- Surface: divers>=1 and oxygen<60 (unit 28) ----------
W1[593, 28] = 6
W1[557, 28] = -12
b1[28] = -3
# ---------- Column recovery (units 29 too right, 30 too left) ----------
W1[42:600:60, 29] = 6
W1[48:600:60, 29] = 6
W1[54:600:60, 29] = 6
b1[29] = -3
W1[0:600:60, 30] = 6
W1[6:600:60, 30] = 6
W1[12:600:60, 30] = 6
W1[18:600:60, 30] = 6
W1[24:600:60, 30] = 6
b1[30] = -3
# ---------- Layer 2: OR groups -> 0/1 features ----------
W2[0:7, 0] = 2
b2[0] = 14
W2[7:14, 1] = 2
b2[1] = 14
W2[14:20, 2] = 2
b2[2] = 12
W2[20:26, 3] = 2
b2[3] = 12
W2[26:28, 4] = 2
b2[4] = 4
W2[28, 5] = 2
b2[5] = 2
W2[29, 6] = 2
b2[6] = 2
W2[30, 7] = 2
b2[7] = 2
# ---------- Output: noop, left, up, right, down, fire ----------
b3[0:6] = [0, 0, -10, 0, 0, 6]
W3[0, 1] = 12
W3[1, 3] = 12
W3[2, 2] = 24
W3[3, 4] = 14
W3[4, 4] = 14
W3[5, 2] = 36
W3[5, 4] = -16
W3[6, 1] = 16
W3[7, 3] = 16
```