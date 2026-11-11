**Design**

The policy keeps the submarine in the two centre columns (cells 4–5). It turns by pressing left or right, which never moves it out of that pair. It fires both ways in alternation. It moves up and down to reach the nearest row with a target, avoids danger, and surfaces when oxygen is low and it carries a diver.

**Layer 1 (sign-like units, |pre-activation| ≥ 8):**
- **Units 0–7, G1..G8:** sub-row thermometer, G_t = [sub row ≥ t].
- **Units 8–15, T1..T8:** row r contains a target (enemy or diver, from the reward channel).
- **Units 16–23, X1..X8:** row r has danger near the centre. This covers any danger in columns 3–6, plus enemy bullets (danger without reward) in columns 1, 2, 7 and 8.
- **Unit 24, SURF:** gauge column 8 lit and column 2 unlit, meaning oxygen < 60 with at least one diver.
- **Unit 25, NODIV:** gauge column 8 unlit, so there is certainly no diver.
- **Units 26–27, CK_R and CK_L:** a sub cell in columns 0–3 (press right) or in columns 6–9 (press left).
- **Units 28–29, TR and TL:** own bullet at column 3, meaning we just fired left, so turn right; own bullet at column 6, meaning we just fired right, so turn left. This gives one shot each way about every 10 steps.

**Layer 2:** each movement unit is gated to the sub's current row s, using G_s and not G_{s+1}.
- **DN_s (s = 0..7) and UP_s (s = 1..8)** compute a score toward the nearest target.
  - Rows at distance d are weighted by powers of 2, so the nearest target dominates, and ties go up.
  - A target in the sub's own row means stay.
  - Danger in the sub's own row (X_s) forces a dodge.
  - SURF forces up and suppresses down.
  - Danger in the row being entered (X_{s±1}) vetoes the move.
- **DN_0** dives from the surface when row 1 is safe.
- **UP_1** (going up from row 1 means surfacing) fires only on SURF, so the sub never surfaces without a diver.
- **UP_2** is blocked by NODIV, unless dodging, which limits the risk of a sticky "up" surfacing with no diver.
- **RIGHT2 and LEFT2** are the "or" of the column-keeping and turn signals.

**Output:** the default action is fire (logit 4). An active vertical unit gives that move logit 10, a turn or column correction gives about 7, and the idle move logits are about −8.

```edits
W1[60:540:6, 0] = 8
W1[120:540:6, 1] = 8
W1[180:540:6, 2] = 8
W1[240:540:6, 3] = 8
W1[300:540:6, 4] = 8
W1[360:540:6, 5] = 8
W1[420:540:6, 6] = 8
W1[480:540:6, 7] = 8
b1[0:8] = -8
W1[62:120:6, 8] = 16
W1[122:180:6, 9] = 16
W1[182:240:6, 10] = 16
W1[242:300:6, 11] = 16
W1[302:360:6, 12] = 16
W1[362:420:6, 13] = 16
W1[422:480:6, 14] = 16
W1[482:540:6, 15] = 16
b1[8:16] = -8
W1[67:115:6, 16] = 16
W1[68:80:6, 16] = -16
W1[104:116:6, 16] = -16
W1[127:175:6, 17] = 16
W1[128:140:6, 17] = -16
W1[164:176:6, 17] = -16
W1[187:235:6, 18] = 16
W1[188:200:6, 18] = -16
W1[224:236:6, 18] = -16
W1[247:295:6, 19] = 16
W1[248:260:6, 19] = -16
W1[284:296:6, 19] = -16
W1[307:355:6, 20] = 16
W1[308:320:6, 20] = -16
W1[344:356:6, 20] = -16
W1[367:415:6, 21] = 16
W1[368:380:6, 21] = -16
W1[404:416:6, 21] = -16
W1[427:475:6, 22] = 16
W1[428:440:6, 22] = -16
W1[464:476:6, 22] = -16
W1[487:535:6, 23] = 16
W1[488:500:6, 23] = -16
W1[524:536:6, 23] = -16
b1[16:24] = -8
W1[593, 24] = 16
W1[557, 24] = -16
b1[24] = -8
W1[593, 25] = -16
b1[25] = 8
W1[0:600:60, 26] = 16
W1[6:600:60, 26] = 16
W1[12:600:60, 26] = 16
W1[18:600:60, 26] = 16
b1[26] = -8
W1[36:600:60, 27] = 16
W1[42:600:60, 27] = 16
W1[48:600:60, 27] = 16
W1[54:600:60, 27] = 16
b1[27] = -8
W1[21:600:60, 28] = 16
b1[28] = -8
W1[39:600:60, 29] = 16
b1[29] = -8
# ---- layer 2: DN_0 (col 0), DN_1..7 (cols 1-7), UP_1..8 (cols 8-15), RIGHT2 (16), LEFT2 (17)
W2[0, 0] = -3000
W2[16, 0] = -2500
W2[0, 1] = 3000
W2[1, 1] = -3000
W2[1, 2] = 3000
W2[2, 2] = -3000
W2[2, 3] = 3000
W2[3, 3] = -3000
W2[3, 4] = 3000
W2[4, 4] = -3000
W2[4, 5] = 3000
W2[5, 5] = -3000
W2[5, 6] = 3000
W2[6, 6] = -3000
W2[6, 7] = 3000
W2[7, 7] = -3000
W2[0, 8] = 3000
W2[1, 8] = -3000
W2[1, 9] = 3000
W2[2, 9] = -3000
W2[2, 10] = 3000
W2[3, 10] = -3000
W2[3, 11] = 3000
W2[4, 11] = -3000
W2[4, 12] = 3000
W2[5, 12] = -3000
W2[5, 13] = 3000
W2[6, 13] = -3000
W2[6, 14] = 3000
W2[7, 14] = -3000
W2[7, 15] = 3000
W2[8:16, 1] = [-256, 128, 64, 32, 16, 8, 4, 2]
W2[8:16, 2] = [-4, -256, 128, 64, 32, 16, 8, 4]
W2[8:16, 3] = [-4, -130, -256, 128, 64, 32, 16, 8]
W2[8:16, 4] = [-4, -66, -130, -256, 128, 64, 32, 16]
W2[8:16, 5] = [-4, -34, -66, -130, -256, 128, 64, 32]
W2[8:16, 6] = [-4, -18, -34, -66, -130, -256, 128, 64]
W2[8:16, 7] = [-4, -10, -18, -34, -66, -130, -256, 128]
W2[8:16, 9] = [130, -256, -128, -64, -32, -16, -8, -4]
W2[8:16, 10] = [4, 130, -256, -128, -64, -32, -16, -8]
W2[8:16, 11] = [4, 66, 130, -256, -128, -64, -32, -16]
W2[8:16, 12] = [4, 34, 66, 130, -256, -128, -64, -32]
W2[8:16, 13] = [4, 18, 34, 66, 130, -256, -128, -64]
W2[8:16, 14] = [4, 10, 18, 34, 66, 130, -256, -128]
W2[8:16, 15] = [4, 6, 10, 18, 34, 66, 130, -256]
W2[16, 1] = 1500
W2[17, 1] = -2500
W2[17, 2] = 1500
W2[18, 2] = -2500
W2[18, 3] = 1500
W2[19, 3] = -2500
W2[19, 4] = 1500
W2[20, 4] = -2500
W2[20, 5] = 1500
W2[21, 5] = -2500
W2[21, 6] = 1500
W2[22, 6] = -2500
W2[22, 7] = 1500
W2[23, 7] = -2500
W2[16, 9] = -2500
W2[17, 9] = 1500
W2[17, 10] = -2500
W2[18, 10] = 1500
W2[18, 11] = -2500
W2[19, 11] = 1500
W2[19, 12] = -2500
W2[20, 12] = 1500
W2[20, 13] = -2500
W2[21, 13] = 1500
W2[21, 14] = -2500
W2[22, 14] = 1500
W2[22, 15] = -2500
W2[23, 15] = 1500
W2[24, 1:8] = -550
W2[24, 8:16] = 550
W2[25, 9] = -500
W2[26, 16] = 2.5
W2[28, 16] = 2.5
W2[27, 16] = -2.5
W2[27, 17] = 2.5
W2[29, 17] = 2.5
W2[26, 17] = -2.5
b2[0:18] = [-5495, -7554, -7560, -7694, -7768, -7818, -7868, -7942, -6000, -7330, -6822, -6748, -6698, -6648, -6574, -3440, 0, 0]
# ---- output: 0 noop, 1 left, 2 up, 3 right, 4 down, 5 fire
W3[0:8, 4] = 9
W3[8:16, 2] = 9
W3[16, 3] = 6.5
W3[17, 1] = 6.5
b3[0:6] = [-5, 0.5, 64, 0.5, 64, 4]
```