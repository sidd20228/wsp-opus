**Design.** The sub stays centred on columns 4–5. It turns left or right to face enemies in its row, fires by default, moves up or down to hunt targets and dodge threats, and surfaces when oxygen is low and it carries a diver.

**Layer 1 (31 units used).** Most units are row-gated. Each has weight 8 on every sub cell in row r and bias −16, so it is live only when the sub is in row r and sits near −1 otherwise. For each row r = 2..8 there are four such units:
- **A_r (turn):** 0.5·(danger in columns 6–9 minus danger in columns 0–3) + 0.5·(trail at column 5 minus trail at column 4). The trail term reads the sub's back cell, so it encodes facing. A value of ±2 or more means "target is behind me, turn". A correction is added when the sub has drifted onto column 3 or 6.
- **V_r (vertical preference):** pull toward rows that hold enemies or divers (adjacent rows 0.7, farther rows 0.3). It adds ±2 for danger near columns 3–6 in the row above or below, so the sub avoids moving into an enemy. Row 2 gets a −1 edge bias and row 8 gets +0.5.
- **C_r (occupancy):** an enemy or diver is in my row.
- **D_r (threat):** an enemy bullet anywhere in my row or an enemy on my own cells gives a high value. An enemy one cell away gives a medium value.

Global units:
- **S:** oxygen-bar column 1 is off and diver-bar column 8 is on, meaning oxygen is below 40 and at least one diver is aboard. This is the signal to surface.
- **CTR:** the sub has drifted left or right of columns 4–5.
- **TOP:** the sub is in rows 0–1, or row 2.

**Layer 2.** Exactly one row unit of each type is live, so layer 2 sums each group and adds a bias that cancels the −1s from the inactive rows. It then thresholds the results into these signals:
- PV (vertical preference)
- turn-right and turn-left
- occupied
- dodge (medium and high levels)
- surface (S)
- centring (CTR)
- no-up in rows 0–2
- rows 0–1

**Output logits.**
- **Fire:** 3 by default, 8 when the row is occupied, and −5 when a turn is needed.
- **Up and down:** ±6·PV, plus a dodge bonus.
- **Surfacing:** up gets +40·S.
- **Rows 0–2:** up is blocked. Rows 0–1 strongly push down.
- **Left and right:** turning plus centring.
- **No-op:** −10.

```edits
b1[0:7] = -16
b1[7] = -17
b1[8:13] = -16
b1[13] = -15.5
b1[14:21] = -17.5
b1[21:28] = -18
b1[28] = -2
b1[30] = -1
# row 2
W1[120:180:6, 0] = 8
W1[120:180:6, 7] = 8
W1[120:180:6, 14] = 8
W1[120:180:6, 21] = 8
W1[138, 0] = 9
W1[156, 0] = 7
W1[157:180:6, 0] = 0.5
W1[121:144:6, 0] = -0.5
W1[154, 0] = 0.5
W1[148, 0] = -0.5
W1[182:540:6, 7] = -0.3
W1[182:240:6, 7] = -0.7
W1[199:218:6, 7] = 2
W1[122:144:6, 14] = 3
W1[158:180:6, 14] = 3
W1[121:180:6, 21] = 4
W1[122:138:6, 21] = -4
W1[164:180:6, 21] = -4
W1[140, 21] = -2
W1[158, 21] = -2
# row 3
W1[180:240:6, 1] = 8
W1[180:240:6, 8] = 8
W1[180:240:6, 15] = 8
W1[180:240:6, 22] = 8
W1[198, 1] = 9
W1[216, 1] = 7
W1[217:240:6, 1] = 0.5
W1[181:204:6, 1] = -0.5
W1[214, 1] = 0.5
W1[208, 1] = -0.5
W1[122:180:6, 8] = 0.7
W1[242:540:6, 8] = -0.3
W1[242:300:6, 8] = -0.7
W1[139:158:6, 8] = -2
W1[259:278:6, 8] = 2
W1[182:204:6, 15] = 3
W1[218:240:6, 15] = 3
W1[181:240:6, 22] = 4
W1[182:198:6, 22] = -4
W1[224:240:6, 22] = -4
W1[200, 22] = -2
W1[218, 22] = -2
# row 4
W1[240:300:6, 2] = 8
W1[240:300:6, 9] = 8
W1[240:300:6, 16] = 8
W1[240:300:6, 23] = 8
W1[258, 2] = 9
W1[276, 2] = 7
W1[277:300:6, 2] = 0.5
W1[241:264:6, 2] = -0.5
W1[274, 2] = 0.5
W1[268, 2] = -0.5
W1[122:240:6, 9] = 0.3
W1[182:240:6, 9] = 0.7
W1[302:540:6, 9] = -0.3
W1[302:360:6, 9] = -0.7
W1[199:218:6, 9] = -2
W1[319:338:6, 9] = 2
W1[242:264:6, 16] = 3
W1[278:300:6, 16] = 3
W1[241:300:6, 23] = 4
W1[242:258:6, 23] = -4
W1[284:300:6, 23] = -4
W1[260, 23] = -2
W1[278, 23] = -2
# row 5
W1[300:360:6, 3] = 8
W1[300:360:6, 10] = 8
W1[300:360:6, 17] = 8
W1[300:360:6, 24] = 8
W1[318, 3] = 9
W1[336, 3] = 7
W1[337:360:6, 3] = 0.5
W1[301:324:6, 3] = -0.5
W1[334, 3] = 0.5
W1[328, 3] = -0.5
W1[122:300:6, 10] = 0.3
W1[242:300:6, 10] = 0.7
W1[362:540:6, 10] = -0.3
W1[362:420:6, 10] = -0.7
W1[259:278:6, 10] = -2
W1[379:398:6, 10] = 2
W1[302:324:6, 17] = 3
W1[338:360:6, 17] = 3
W1[301:360:6, 24] = 4
W1[302:318:6, 24] = -4
W1[344:360:6, 24] = -4
W1[320, 24] = -2
W1[338, 24] = -2
# row 6
W1[360:420:6, 4] = 8
W1[360:420:6, 11] = 8
W1[360:420:6, 18] = 8
W1[360:420:6, 25] = 8
W1[378, 4] = 9
W1[396, 4] = 7
W1[397:420:6, 4] = 0.5
W1[361:384:6, 4] = -0.5
W1[394, 4] = 0.5
W1[388, 4] = -0.5
W1[122:360:6, 11] = 0.3
W1[302:360:6, 11] = 0.7
W1[422:540:6, 11] = -0.3
W1[422:480:6, 11] = -0.7
W1[319:338:6, 11] = -2
W1[439:458:6, 11] = 2
W1[362:384:6, 18] = 3
W1[398:420:6, 18] = 3
W1[361:420:6, 25] = 4
W1[362:378:6, 25] = -4
W1[404:420:6, 25] = -4
W1[380, 25] = -2
W1[398, 25] = -2
# row 7
W1[420:480:6, 5] = 8
W1[420:480:6, 12] = 8
W1[420:480:6, 19] = 8
W1[420:480:6, 26] = 8
W1[438, 5] = 9
W1[456, 5] = 7
W1[457:480:6, 5] = 0.5
W1[421:444:6, 5] = -0.5
W1[454, 5] = 0.5
W1[448, 5] = -0.5
W1[122:420:6, 12] = 0.3
W1[362:420:6, 12] = 0.7
W1[482:540:6, 12] = -0.7
W1[379:398:6, 12] = -2
W1[499:518:6, 12] = 2
W1[422:444:6, 19] = 3
W1[458:480:6, 19] = 3
W1[421:480:6, 26] = 4
W1[422:438:6, 26] = -4
W1[464:480:6, 26] = -4
W1[440, 26] = -2
W1[458, 26] = -2
# row 8
W1[480:540:6, 6] = 8
W1[480:540:6, 13] = 8
W1[480:540:6, 20] = 8
W1[480:540:6, 27] = 8
W1[498, 6] = 9
W1[516, 6] = 7
W1[517:540:6, 6] = 0.5
W1[481:504:6, 6] = -0.5
W1[514, 6] = 0.5
W1[508, 6] = -0.5
W1[122:480:6, 13] = 0.3
W1[422:480:6, 13] = 0.7
W1[439:458:6, 13] = -2
W1[482:504:6, 20] = 3
W1[518:540:6, 20] = 3
W1[481:540:6, 27] = 4
W1[482:498:6, 27] = -4
W1[524:540:6, 27] = -4
W1[500, 27] = -2
W1[518, 27] = -2
# globals: surface-needed, centring, top rows
W1[593, 28] = 4
W1[551, 28] = -4
W1[36:600:60, 29] = 2
W1[42:600:60, 29] = 2
W1[48:600:60, 29] = 2
W1[54:600:60, 29] = 2
W1[0:600:60, 29] = -2
W1[6:600:60, 29] = -2
W1[12:600:60, 29] = -2
W1[18:600:60, 29] = -2
W1[0:120:6, 30] = 2
W1[120:180:6, 30] = 1
# layer 2
W2[7:14, 0] = 2
b2[0] = 12
W2[0:7, 1] = 15
b2[1] = 81
W2[0:7, 2] = -15
b2[2] = -99
W2[14:21, 3] = 3
b2[3] = 18
W2[21:28, 4] = 4
b2[4] = 26
W2[21:28, 5] = 4
b2[5] = 22
W2[28, 6] = 3
W2[29, 7] = 2
W2[30, 8] = 10
W2[30, 9] = 30
b2[9] = -26.4
# outputs: 0 noop, 1 left, 2 up, 3 right, 4 down, 5 fire
b3[0] = -10
W3[3, 5] = 2.5
W3[2, 5] = -2.5
W3[1, 5] = -2.5
b3[5] = 0.5
W3[0, 2] = 6
W3[4, 2] = 1.5
W3[5, 2] = 5
W3[8, 2] = -8
W3[6, 2] = 20
b3[2] = 17.5
W3[0, 4] = -6
W3[4, 4] = 1.5
W3[5, 4] = 5
W3[9, 4] = 6
W3[6, 4] = -20
b3[4] = -8.5
W3[2, 1] = 4
W3[7, 1] = 6
W3[4, 1] = -1.5
W3[9, 1] = -7
b3[1] = -6.5
W3[1, 3] = 4
W3[7, 3] = -6
W3[4, 3] = -1.5
W3[9, 3] = -7
b3[3] = -6.5
```