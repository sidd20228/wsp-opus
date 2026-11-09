**Strategy: a stationary gunner at column 8 that moves only vertically**

1. **Park at column 8, facing left.** At the surface (row 0), where nothing can hurt the submarine and oxygen does not drain, it presses right until it reaches column 9, then presses left once. That leaves the front at column 8, facing left. Underwater it never moves sideways, so "my column" is fixed and every bullet sweeps columns 7 to 0.
2. **Wait at the surface, then dive.** It stays on the surface until a target appears in rows 2 to 8 and row 1 is safe, then dives. Row 1 is never targeted, because a sticky repeated "up" from row 1 would surface with no diver and end the game.
3. **Move to the nearest target row.** Targets are:
   - enemies anywhere in the row;
   - divers that will drift into column 8 (moving right in columns 4 to 7, or just spawned at column 9).

   Direction of travel is read from the trail channel. When the submarine reaches a row with a target, it stays and fires; fire is the default action.
4. **Avoid danger.** Danger means anything in the danger channel at columns 6 to 9 of a row.
   - It never moves into a dangerous row.
   - If its own row becomes dangerous, it flees: up from rows 5 to 8, down from rows 1 to 4, and the other way if that row is blocked.
5. **Surface for oxygen.** When oxygen is below 60 (gauge cell (9,2) off) and at least one diver is aboard (cell (9,8) on, which is unambiguous at that oxygen level), it heads straight up, still respecting danger, refills, and dives again.

**Layer 1 (27 of 32 units used)**
- **Units 0–8, Z_s:** the submarine is in row s.
- **Units 9–15, T_r (rows 2–8):** the row has a target. Each reward cell is weighted 24+12c, and its trail cell subtracts 12(c−1) or 12(c+1). This gives right-moving objects +12 and left-moving ones −12; the danger channel adds 12 so enemies always count. Divers in columns 0–3 do not count, and a lone enemy bullet does not count. The submarine's own back cell is cancelled out.
- **Units 16–23, W_r (rows 1–8):** danger in columns 6–9 of that row.
- **Unit 24, SM:** surface mode (low oxygen and a diver aboard).
- **Unit 25, H0:** at the surface in the parking spot.
- **Unit 26, N9:** at the surface, touching column 9.

**Layer 2 (27 units used)**
- **UP_s and DOWN_s, one per row:** each is a single gated threshold unit. It is active only when the submarine is in row s, and it is vetoed when the destination row is dangerous.
- **How UP_s and DOWN_s weigh the evidence:**
  - Distance-weighted target sums decide direction: targets above count +126, +78, +54, … by distance, and targets below count −120, −72, −48, …. Nearer targets dominate and the direction cannot oscillate.
  - A target in the submarine's own row means stay.
  - Danger in its own row means flee.
  - Surface mode means go up.
- **DOWN_0:** dives from the surface.
- **LEFT_0 / RIGHT_0:** the parking moves at the surface.
- **Fallback units:** flee the other way when the preferred escape row is blocked.

**Outputs**
- Up, down, left and right logits sum their layer-2 units; each is about +10 (+15 for left/right) when a unit is active and −10 (−15) when none is.
- Fire is fixed at 0, so it wins whenever no movement is requested.
- No-op is −20 and is never chosen.

```edits
# ---------- Layer 1 ----------
# Z_s: sub in row s (units 0..8)
W1[0:60:6, 0] = 6
W1[60:120:6, 1] = 6
W1[120:180:6, 2] = 6
W1[180:240:6, 3] = 6
W1[240:300:6, 4] = 6
W1[300:360:6, 5] = 6
W1[360:420:6, 6] = 6
W1[420:480:6, 7] = 6
W1[480:540:6, 8] = 6
b1[0:9] = -6
# T_r: target in row r (r=2..8 -> units 9..15)
W1[122:180:6, 9] = [12,12,12,12,24,36,48,60,72,84]
W1[121:180:6, 9] = 12
W1[154:180:6, 9] = [-12,-24,-36,-48,-60]
W1[174, 9] = 60
W1[182:240:6, 10] = [12,12,12,12,24,36,48,60,72,84]
W1[181:240:6, 10] = 12
W1[214:240:6, 10] = [-12,-24,-36,-48,-60]
W1[234, 10] = 60
W1[242:300:6, 11] = [12,12,12,12,24,36,48,60,72,84]
W1[241:300:6, 11] = 12
W1[274:300:6, 11] = [-12,-24,-36,-48,-60]
W1[294, 11] = 60
W1[302:360:6, 12] = [12,12,12,12,24,36,48,60,72,84]
W1[301:360:6, 12] = 12
W1[334:360:6, 12] = [-12,-24,-36,-48,-60]
W1[354, 12] = 60
W1[362:420:6, 13] = [12,12,12,12,24,36,48,60,72,84]
W1[361:420:6, 13] = 12
W1[394:420:6, 13] = [-12,-24,-36,-48,-60]
W1[414, 13] = 60
W1[422:480:6, 14] = [12,12,12,12,24,36,48,60,72,84]
W1[421:480:6, 14] = 12
W1[454:480:6, 14] = [-12,-24,-36,-48,-60]
W1[474, 14] = 60
W1[482:540:6, 15] = [12,12,12,12,24,36,48,60,72,84]
W1[481:540:6, 15] = 12
W1[514:540:6, 15] = [-12,-24,-36,-48,-60]
W1[534, 15] = 60
b1[9:16] = -18
# W_r: danger in cols 6..9 of row r (r=1..8 -> units 16..23)
W1[97:120:6, 16] = 12
W1[157:180:6, 17] = 12
W1[217:240:6, 18] = 12
W1[277:300:6, 19] = 12
W1[337:360:6, 20] = 12
W1[397:420:6, 21] = 12
W1[457:480:6, 22] = 12
W1[517:540:6, 23] = 12
b1[16:24] = -6
# SM: oxygen<60 and diver aboard
W1[593, 24] = 12
W1[557, 24] = -12
b1[24] = -6
# H0: parked at surface (back at (0,9)); N9: sub touches (0,9)
W1[58, 25] = 12
b1[25] = -6
W1[54, 26] = 12
b1[26] = -6

# ---------- Layer 2 ----------
# UP_1 (unit 0): only in surface mode
W2[1, 0] = 6
W2[24, 0] = 3
b2[0] = -6
# UP_2..UP_4 (units 1..3)
W2[2, 1] = 1200
W2[16:18, 1] = [-1200, -200]
W2[9:16, 1] = [-200,-60,-36,-24,-18,-15,-12]
b2[1] = -2368
W2[3, 2] = 1200
W2[17:19, 2] = [-1200, -200]
W2[9:16, 2] = [63,-200,-60,-36,-24,-18,-15]
b2[2] = -2293
W2[4, 3] = 1200
W2[18:20, 3] = [-1200, -200]
W2[9:16, 3] = [39,63,-200,-60,-36,-24,-18]
b2[3] = -2239
W2[24, 1:4] = 600
# UP_5..UP_8 (units 4..7)
W2[5, 4] = 1000
W2[19:21, 4] = [-1000, 400]
W2[9:16, 4] = [27,39,63,-200,-60,-36,-24]
b2[4] = -1394
W2[6, 5] = 1000
W2[20:22, 5] = [-1000, 400]
W2[9:16, 5] = [21,27,39,63,-200,-60,-36]
b2[5] = -1349
W2[7, 6] = 1000
W2[21:23, 6] = [-1000, 400]
W2[9:16, 6] = [18,21,27,39,63,-200,-60]
b2[6] = -1295
W2[8, 7] = 1000
W2[22:24, 7] = [-1000, 400]
W2[9:16, 7] = [15,18,21,27,39,63,-200]
b2[7] = -1220
W2[24, 4:8] = 400
# DOWN_1..DOWN_4 (units 8..11)
W2[1, 8] = 1200
W2[16:18, 8] = [400, -1200]
W2[9:16, 8] = [60,36,24,18,15,12,9]
b2[8] = -2429
W2[2, 9] = 1200
W2[17:19, 9] = [400, -1200]
W2[9:16, 9] = [-200,60,36,24,18,15,12]
b2[9] = -2638
W2[3, 10] = 1200
W2[18:20, 10] = [400, -1200]
W2[9:16, 10] = [-63,-200,60,36,24,18,15]
b2[10] = -2713
W2[4, 11] = 1200
W2[19:21, 11] = [400, -1200]
W2[9:16, 11] = [-39,-63,-200,60,36,24,18]
b2[11] = -2767
W2[24, 8:12] = -600
# DOWN_5..DOWN_7 (units 12..14)
W2[5, 12] = 200
W2[20:22, 12] = -200
W2[9:16, 12] = [-27,-39,-63,-200,60,36,24]
b2[12] = -1012
W2[6, 13] = 200
W2[21:23, 13] = -200
W2[9:16, 13] = [-21,-27,-39,-63,-200,60,36]
b2[13] = -1057
W2[7, 14] = 200
W2[22:24, 14] = -200
W2[9:16, 14] = [-18,-21,-27,-39,-63,-200,60]
b2[14] = -1111
W2[24, 12:15] = -200
# DOWN_0 (unit 15): dive when parked, targets exist, row 1 safe
W2[0, 15] = 30
W2[25, 15] = 30
W2[16, 15] = -60
W2[9:16, 15] = 3
b2[15] = -102
# LEFT_0 (unit 16), RIGHT_0 (unit 17): parking at surface
W2[0, 16] = 3
W2[26, 16] = 3
W2[25, 16] = -6
b2[16] = -9
W2[0, 17] = 3
W2[26, 17] = -3
b2[17] = -3
# FBU_2..4 (units 18..20): flee up when own and lower rows dangerous
W2[2, 18] = 3
W2[16:19, 18] = [-6, 3, 3]
W2[3, 19] = 3
W2[17:20, 19] = [-6, 3, 3]
W2[4, 20] = 3
W2[18:21, 20] = [-6, 3, 3]
b2[18:21] = -12
# FBD_2..7 (units 21..26): flee down when own and upper rows dangerous
W2[2, 21] = 3
W2[16:19, 21] = [3, 3, -6]
W2[3, 22] = 3
W2[17:20, 22] = [3, 3, -6]
W2[4, 23] = 3
W2[18:21, 23] = [3, 3, -6]
W2[5, 24] = 3
W2[19:22, 24] = [3, 3, -6]
W2[6, 25] = 3
W2[20:23, 25] = [3, 3, -6]
W2[7, 26] = 3
W2[21:24, 26] = [3, 3, -6]
b2[21:27] = -12

# ---------- Output ----------
W3[0:8, 2] = 10
W3[18:21, 2] = 10
W3[8:16, 4] = 10
W3[21:27, 4] = 10
W3[16, 1] = 15
W3[17, 3] = 15
b3[0:6] = [-20, 0, 100, 0, 130, 0]
```