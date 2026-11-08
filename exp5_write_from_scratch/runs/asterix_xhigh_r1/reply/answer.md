# Hand-Written Policy for MinAtar Asterix: Fixed-Column Vertical Dodger with Gold Seeking

## Strategy

**Never move sideways.** The left, right and fire logits are about 170 below the others, so they are never sampled. Sticky actions only repeat earlier actions, so they cannot move the player sideways either. The player therefore stays in column 5 for the whole game.

This makes every threat a fixed pattern of cells in each row. The game also allows at most one entity per row, so each row's danger and treasure state can be read directly from that row's cells.

**Direction.** An entity's direction comes from its trail cell:
- If the trail is at column c+1, the entity is moving left.
- If the trail is at column c−1, the entity is moving right.

**Each step, the network compares three options:**
- staying in the current row,
- moving up (into row y−1),
- moving down (into row y+1).

Each option gets a value:
- **Stay:** +0.4 bonus, −3 if an enemy is one step away and approaching, −1.5 if it is two steps away and approaching, +3 if treasure is approaching.
- **Enter a neighbouring row:** −3 if an enemy is in column 5 or one step away and approaching (deadly). −1.5 if an enemy is two steps away and approaching. +1.3 if treasure is approaching or in column 5. Smaller bonuses (0.9, 0.75, 0.6 by distance) for approaching treasure in rows further in that direction.
- **Edge rows:** a small "comfort" penalty discourages rows 1 and 8, because they have only one escape route.
- **Top and bottom edges:** in row 1, "up" means staying put, so up gets the stay value. Row 8 works the same way for "down".

## Hidden Units

**Layer 1 (32 units, each a saturated ±1 threshold unit):**

| Units | Name | Fires when (in row r = 1..8) |
|---|---|---|
| 0–7 | P_y | The player is in row y. |
| 8–15 | D1_r | An enemy is in column 5, or in column 6 with its trail in column 7, or in column 4 with its trail in column 3. This is an AND of an enemy cell and a trail cell. |
| 16–23 | D2_r | An enemy is in column 7 with its trail in column 8, or in column 3 with its trail in column 2. |
| 24–31 | G_r | Treasure is approaching column 5 (at most 4 cells away) or sitting in column 5. Gold weights increase toward the centre and trail weights mark the "far side". Any enemy in the row vetoes the unit with weight −60. |

**Layer 2 (22 gated units, the rest unused):**
- **Units 0–7 (S_y):** the stay value for row y.
- **Units 8–14 (U_y):** the value of moving up from row y.
- **Units 15–21 (Dn_y):** the value of moving down from row y.

Each layer-2 unit computes tanh(6·P_y − 6 + 0.5·value).
- If the player is not in row y, the unit sits at exactly −1. That gives the same constant offset to every logit, which b3 cancels.
- If the player is in row y, the unit outputs tanh(0.5·value).

**Logits:** each of no-op, up and down is 30 × (the sum of its units) + 210.

```edits
# ===== Layer 1: player row units P_y (y=1..8 -> unit y-1) =====
W1[60:120:6, 0] = 6
W1[120:180:6, 1] = 6
W1[180:240:6, 2] = 6
W1[240:300:6, 3] = 6
W1[300:360:6, 4] = 6
W1[360:420:6, 5] = 6
W1[420:480:6, 6] = 6
W1[480:540:6, 7] = 6
b1[0:8] = -3
# ===== D1_r (unit 7+r): enemy at col5, or col6 moving left, or col4 moving right =====
W1[91, 8] = 18
W1[97, 8] = 12
W1[85, 8] = 12
W1[106, 8] = 6
W1[82, 8] = 6
W1[151, 9] = 18
W1[157, 9] = 12
W1[145, 9] = 12
W1[166, 9] = 6
W1[142, 9] = 6
W1[211, 10] = 18
W1[217, 10] = 12
W1[205, 10] = 12
W1[226, 10] = 6
W1[202, 10] = 6
W1[271, 11] = 18
W1[277, 11] = 12
W1[265, 11] = 12
W1[286, 11] = 6
W1[262, 11] = 6
W1[331, 12] = 18
W1[337, 12] = 12
W1[325, 12] = 12
W1[346, 12] = 6
W1[322, 12] = 6
W1[391, 13] = 18
W1[397, 13] = 12
W1[385, 13] = 12
W1[406, 13] = 6
W1[382, 13] = 6
W1[451, 14] = 18
W1[457, 14] = 12
W1[445, 14] = 12
W1[466, 14] = 6
W1[442, 14] = 6
W1[511, 15] = 18
W1[517, 15] = 12
W1[505, 15] = 12
W1[526, 15] = 6
W1[502, 15] = 6
b1[8:16] = -15
# ===== D2_r (unit 15+r): enemy at col7 moving left, or col3 moving right =====
W1[103, 16] = 12
W1[79, 16] = 12
W1[112, 16] = 6
W1[76, 16] = 6
W1[163, 17] = 12
W1[139, 17] = 12
W1[172, 17] = 6
W1[136, 17] = 6
W1[223, 18] = 12
W1[199, 18] = 12
W1[232, 18] = 6
W1[196, 18] = 6
W1[283, 19] = 12
W1[259, 19] = 12
W1[292, 19] = 6
W1[256, 19] = 6
W1[343, 20] = 12
W1[319, 20] = 12
W1[352, 20] = 6
W1[316, 20] = 6
W1[403, 21] = 12
W1[379, 21] = 12
W1[412, 21] = 6
W1[376, 21] = 6
W1[463, 22] = 12
W1[439, 22] = 12
W1[472, 22] = 6
W1[436, 22] = 6
W1[523, 23] = 12
W1[499, 23] = 12
W1[532, 23] = 6
W1[496, 23] = 6
b1[16:24] = -15
# ===== G_r (unit 23+r): treasure approaching column 5 =====
W1[62:122:6, 24] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[64:124:6, 24] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[61:121:6, 24] = -60
W1[122:182:6, 25] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[124:184:6, 25] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[121:181:6, 25] = -60
W1[182:242:6, 26] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[184:244:6, 26] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[181:241:6, 26] = -60
W1[242:302:6, 27] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[244:304:6, 27] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[241:301:6, 27] = -60
W1[302:362:6, 28] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[304:364:6, 28] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[301:361:6, 28] = -60
W1[362:422:6, 29] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[364:424:6, 29] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[361:421:6, 29] = -60
W1[422:482:6, 30] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[424:484:6, 30] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[421:481:6, 30] = -60
W1[482:542:6, 31] = [0, 0, 6, 12, 18, 24, 18, 12, 6, 0]
W1[484:544:6, 31] = [27, 18, 15, 9, 3, 0, 3, 9, 15, 18]
W1[481:541:6, 31] = -60
b1[24:32] = -21
# ===== Layer 2: STAY units S_y (h2 unit y-1) =====
W2[0, 0] = 6
W2[1, 1] = 6
W2[2, 2] = 6
W2[3, 3] = 6
W2[4, 4] = 6
W2[5, 5] = 6
W2[6, 6] = 6
W2[7, 7] = 6
W2[8, 0] = -0.75
W2[9, 1] = -0.75
W2[10, 2] = -0.75
W2[11, 3] = -0.75
W2[12, 4] = -0.75
W2[13, 5] = -0.75
W2[14, 6] = -0.75
W2[15, 7] = -0.75
W2[16, 0] = -0.375
W2[17, 1] = -0.375
W2[18, 2] = -0.375
W2[19, 3] = -0.375
W2[20, 4] = -0.375
W2[21, 5] = -0.375
W2[22, 6] = -0.375
W2[23, 7] = -0.375
W2[24, 0] = 0.75
W2[25, 1] = 0.75
W2[26, 2] = 0.75
W2[27, 3] = 0.75
W2[28, 4] = 0.75
W2[29, 5] = 0.75
W2[30, 6] = 0.75
W2[31, 7] = 0.75
b2[0:8] = [-6.5, -6.275, -6.2, -6.175, -6.175, -6.2, -6.275, -6.5]
# ===== UP units U_y (y=2..8 -> h2 unit 6+y), target row y-1 =====
W2[1, 8] = 6
W2[2, 9] = 6
W2[3, 10] = 6
W2[4, 11] = 6
W2[5, 12] = 6
W2[6, 13] = 6
W2[7, 14] = 6
W2[8, 8] = -0.75
W2[9, 9] = -0.75
W2[10, 10] = -0.75
W2[11, 11] = -0.75
W2[12, 12] = -0.75
W2[13, 13] = -0.75
W2[14, 14] = -0.75
W2[16, 8] = -0.375
W2[17, 9] = -0.375
W2[18, 10] = -0.375
W2[19, 11] = -0.375
W2[20, 12] = -0.375
W2[21, 13] = -0.375
W2[22, 14] = -0.375
W2[24:31, 8:15] = [[0.325, 0.225, 0.1875, 0.15, 0.15, 0.15, 0.15],
  [0, 0.325, 0.225, 0.1875, 0.15, 0.15, 0.15],
  [0, 0, 0.325, 0.225, 0.1875, 0.15, 0.15],
  [0, 0, 0, 0.325, 0.225, 0.1875, 0.15],
  [0, 0, 0, 0, 0.325, 0.225, 0.1875],
  [0, 0, 0, 0, 0, 0.325, 0.225],
  [0, 0, 0, 0, 0, 0, 0.325]]
b2[8:15] = [-7.125, -6.675, -6.4125, -6.2375, -6.0875, -5.9625, -5.8875]
# ===== DOWN units Dn_y (y=1..7 -> h2 unit 14+y), target row y+1 =====
W2[0, 15] = 6
W2[1, 16] = 6
W2[2, 17] = 6
W2[3, 18] = 6
W2[4, 19] = 6
W2[5, 20] = 6
W2[6, 21] = 6
W2[9, 15] = -0.75
W2[10, 16] = -0.75
W2[11, 17] = -0.75
W2[12, 18] = -0.75
W2[13, 19] = -0.75
W2[14, 20] = -0.75
W2[15, 21] = -0.75
W2[17, 15] = -0.375
W2[18, 16] = -0.375
W2[19, 17] = -0.375
W2[20, 18] = -0.375
W2[21, 19] = -0.375
W2[22, 20] = -0.375
W2[23, 21] = -0.375
W2[25:32, 15:22] = [[0.325, 0, 0, 0, 0, 0, 0],
  [0.225, 0.325, 0, 0, 0, 0, 0],
  [0.1875, 0.225, 0.325, 0, 0, 0, 0],
  [0.15, 0.1875, 0.225, 0.325, 0, 0, 0],
  [0.15, 0.15, 0.1875, 0.225, 0.325, 0, 0],
  [0.15, 0.15, 0.15, 0.1875, 0.225, 0.325, 0],
  [0.15, 0.15, 0.15, 0.15, 0.1875, 0.225, 0.325]]
b2[15:22] = [-5.8875, -5.9625, -6.0875, -6.2375, -6.4125, -6.675, -7.125]
# ===== Output =====
W3[0:8, 0] = 30
W3[8:15, 2] = 30
W3[0, 2] = 30
W3[15:22, 4] = 30
W3[7, 4] = 30
b3[0:6] = [210, -200, 210, -200, 210, -200]
```