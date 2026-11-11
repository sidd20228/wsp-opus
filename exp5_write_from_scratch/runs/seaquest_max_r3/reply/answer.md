## Design

**Strategy: an edge turret.** The submarine parks at column 8 facing left, with its back cell at column 9. From there it never moves sideways again, only up and down, and it fires whenever it isn't moving.

- **Shooting:** facing left, its bullets sweep columns 0–7 of its row, so every enemy in front of it can be shot. Enemies always move slower than bullets.
- **Collisions:** only the front cell counts, so a fish that spawns at column 9 is harmless for about 5 steps. That is enough time to change rows.
- **Divers:** divers coming from either side reach column 8 and are picked up automatically.

**Startup.** The setup happens at row 0, which has no enemies and doesn't use oxygen. At row 0 the trail channel holds only the sub's own back cell, so orientation can be read exactly:
- If the sub doesn't cover column 9, press right.
- If the front is at column 9, press left.
- Once the back cell is at column 9 (home), wait at the surface until a target appears, then dive.

**Hidden layer 1** (unit 3(r−1)+k for rows r = 1..8). Each unit is gated by "sub is in row r": the 2 player cells give +1000 each, and the bias is about −2000.
- **U_r (go up):**
  - Leave: +300 for a danger at column 9 of my row, or an enemy bullet at columns 3–7.
  - Stay: −60 for each target in my row.
  - Seek: +24 for targets in the next row up, +10 further up; −24 / −10 for targets below.
  - Veto: −1500 if the row above has danger at columns 6–9, or the row two above has danger at columns 7–9 (protection against a sticky repeated move).
  - U_1 and U_2 are switched off, so the sub never deliberately rises into row 1 and risks surfacing with no diver.
- **D_r (go down):** the mirror image of U_r. D_8 is switched off.
- **SU_r:** "sub in row r and the row above is safe". It is used only for surfacing.
- **G** (unit 24): oxygen gauge cell 2 is off (oxygen < 60) and diver gauge cell 8 is on (at least 1 diver).
- **Row-0 units:** press right (25), press left (26), and descend (27; needs home + any target + row 1 safe).

**Layer 2 and outputs.**
- Layer 2 computes UP = OR(U), DOWN = OR(D, descend), GUP = G AND OR(SU), and copies of the left/right units.
- Logits: fire is the default (10). Up gets 30, or 50 when surfacing is needed. Down and left/right get 30. Inactive actions sit at −20.

```edits
b1[0:24] = -2003
b1[1:24:3] = -1997
b1[0] = -100000
b1[3] = -100000
b1[23] = -100000
b1[24] = -5
b1[25] = -10
b1[26] = -5
b1[27] = -1005
W1[60:120:6, 0:3] = 1000
W1[62:120:6, 0:3:2] = -60
W1[80:105:6, 0:3:2] = -360
W1[79:120:6, 0:3:2] = 300
W1[120:180:6, 3:6] = 1000
W1[122:180:6, 3:6:2] = -60
W1[140:165:6, 3:6:2] = -360
W1[139:180:6, 3:6:2] = 300
W1[180:240:6, 6:9] = 1000
W1[182:240:6, 6:9:2] = -60
W1[200:225:6, 6:9:2] = -360
W1[199:240:6, 6:9:2] = 300
W1[240:300:6, 9:12] = 1000
W1[242:300:6, 9:12:2] = -60
W1[260:285:6, 9:12:2] = -360
W1[259:300:6, 9:12:2] = 300
W1[300:360:6, 12:15] = 1000
W1[302:360:6, 12:15:2] = -60
W1[320:345:6, 12:15:2] = -360
W1[319:360:6, 12:15:2] = 300
W1[360:420:6, 15:18] = 1000
W1[362:420:6, 15:18:2] = -60
W1[380:405:6, 15:18:2] = -360
W1[379:420:6, 15:18:2] = 300
W1[420:480:6, 18:21] = 1000
W1[422:480:6, 18:21:2] = -60
W1[440:465:6, 18:21:2] = -360
W1[439:480:6, 18:21:2] = 300
W1[480:540:6, 21:24] = 1000
W1[482:540:6, 21:24:2] = -60
W1[500:525:6, 21:24:2] = -360
W1[499:540:6, 21:24:2] = 300
# up-safety (U and SU)
W1[97:120:6, 3:5] = -1500
W1[157:180:6, 6:8] = -1500
W1[103:120:6, 6:8] = -1500
W1[217:240:6, 9:11] = -1500
W1[163:180:6, 9:11] = -1500
W1[277:300:6, 12:14] = -1500
W1[223:240:6, 12:14] = -1500
W1[337:360:6, 15:17] = -1500
W1[283:300:6, 15:17] = -1500
W1[397:420:6, 18:20] = -1500
W1[343:360:6, 18:20] = -1500
W1[457:480:6, 21:23] = -1500
W1[403:420:6, 21:23] = -1500
# down-safety (D)
W1[157:180:6, 2] = -1500
W1[223:240:6, 2] = -1500
W1[217:240:6, 5] = -1500
W1[283:300:6, 5] = -1500
W1[277:300:6, 8] = -1500
W1[343:360:6, 8] = -1500
W1[337:360:6, 11] = -1500
W1[403:420:6, 11] = -1500
W1[397:420:6, 14] = -1500
W1[463:480:6, 14] = -1500
W1[457:480:6, 17] = -1500
W1[523:540:6, 17] = -1500
W1[517:540:6, 20] = -1500
# seek: U units
W1[122:180:6, 6] = 24
W1[242:300:6, 6] = -24
W1[302:540:6, 6] = -10
W1[182:240:6, 9] = 24
W1[122:180:6, 9] = 10
W1[302:360:6, 9] = -24
W1[362:540:6, 9] = -10
W1[242:300:6, 12] = 24
W1[122:240:6, 12] = 10
W1[362:420:6, 12] = -24
W1[422:540:6, 12] = -10
W1[302:360:6, 15] = 24
W1[122:300:6, 15] = 10
W1[422:480:6, 15] = -24
W1[482:540:6, 15] = -10
W1[362:420:6, 18] = 24
W1[122:360:6, 18] = 10
W1[482:540:6, 18] = -24
W1[422:480:6, 21] = 24
W1[122:420:6, 21] = 10
# seek: D units
W1[122:180:6, 2] = 24
W1[182:540:6, 2] = 10
W1[182:240:6, 5] = 24
W1[242:540:6, 5] = 10
W1[242:300:6, 8] = 24
W1[302:540:6, 8] = 10
W1[122:180:6, 8] = -24
W1[302:360:6, 11] = 24
W1[362:540:6, 11] = 10
W1[182:240:6, 11] = -24
W1[122:180:6, 11] = -10
W1[362:420:6, 14] = 24
W1[422:540:6, 14] = 10
W1[242:300:6, 14] = -24
W1[122:240:6, 14] = -10
W1[422:480:6, 17] = 24
W1[482:540:6, 17] = 10
W1[302:360:6, 17] = -24
W1[122:300:6, 17] = -10
W1[482:540:6, 20] = 24
W1[362:420:6, 20] = -24
W1[122:360:6, 20] = -10
# G, row-0 setup, descend
W1[593, 24] = 10
W1[557, 24] = -10
W1[0:60:6, 25] = 10
W1[54, 25] = -10
W1[54, 26] = 10
W1[58, 26] = -10
W1[58, 27] = 1000
W1[62:540:6, 27] = 10
W1[97:120:6, 27] = -1500
W1[163:180:6, 27] = -1500
# layer 2
W2[0:24:3, 0] = 5
b2[0] = 35
W2[2:24:3, 1] = 5
W2[27, 1] = 5
b2[1] = 40
W2[1:24:3, 2] = 5
W2[24, 2] = 5
b2[2] = 30
W2[25, 3] = 5
W2[26, 4] = 5
# outputs
W3[0, 2] = 25
W3[2, 2] = 35
b3[2] = 40
W3[1, 4] = 25
b3[4] = 5
W3[3, 3] = 25
b3[3] = 5
W3[4, 1] = 25
b3[1] = 5
b3[5] = 10
b3[0] = -20
```