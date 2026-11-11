**Observations**

- **Frame.** The W1 rows for row 0 (pixels 0–11), column 0 (pixels 12, 24, …, 132) and column 11 (pixels 23, 35, …, 143) are all |w| ≲ 0.03. Their b3 values are about −0.33 to −0.54 (b3[0] = −0.457, b3[36] = −0.535, b3[143] = −0.456). These pixels never vary and are always black.
- **Active pixels form an envelope that is narrow at the top and wide at the bottom.**
  - Row 1: only columns 3–8 matter (p15 0.112, p16 0.105, p17 0.389, p18 0.173, p19 0.157, p20 0.176).
  - Row 2: columns 2–9 (p26 0.088, p29 0.629, p30 0.426, p32 0.396, p33 0.138).
  - Row 3: columns 2–9 (p38 0.258, p45 0.323; p37 and p46 are tiny).
  - Row 4 down to row 11: columns 1–10 (p49 0.121, p58 0.157, p61 0.238, row 11 into unit 38 at 0.129 … 0.326 … 0.143).
  - The top-corner pixels (p13, p14, p21, p22, p25, p37) are inactive.
- **b3 is itself a soft triangle.**
  - Row 1: (1,5) = −0.143 and (1,6) = −0.105, versus about −0.30 at (1,4) and (1,7).
  - Row 2: (2,5) to (2,7) are about −0.01 to −0.06.
  - Rows 4–9 interior: about −0.08 to +0.06.
  - Border: about −0.45.
- **Layer-1 detectors.**
  - Apex blob, unit 19: p17 0.389, p29 0.629, p40 0.299, p41 0.349, p42 0.271, p52 0.277.
  - Left-side bands:
    - unit 4: p27 .314, p39 .382, p50 .405, p62 .331, p73 .202
    - unit 5: p28 .338, p51 .416, p74 .398, p85 .261
    - unit 18: p86 .313, p98 .435, p109 .432
  - Right-side bands:
    - unit 14: p32 .396, p44 .391, p57 .421
    - unit 20: p56 .400, p69 .371, p81 .514, p94 .395
    - unit 59: p105 .460, p118 .454
  - Horizontal rows:
    - unit 8: row 10 (p125–129: .442 .453 .483 .482 .411), with same-sign weights on row 9 (.26–.31) and row 8 (.09–.15)
    - unit 44: row 9 (p111–116: .305–.361) plus row 8 (.11–.17)
    - unit 38: row 11
    - unit 17: row 10 negative (−.26 to −.36)
    - unit 47: row 7 negative (p88–91: −.39 to −.43)
    - units 12/13: row 8 negative (−.30 to −.36)
  - Half-plane edge detectors:
    - unit 6 is negative along p54 −.401, p65 −.413, p77 −.488, p88 −.416, p100 −.401, and positive to the right (p91 .159, p103 .151).
    - unit 27 is negative along p65 −.398, p78 −.336, p90 −.416, p103 −.386, and positive to the left (p100 .178, p111 .178, p112 .182).
- **Time inputs.** Only the k = 4..7 sin/cos rows are large (row 157: −1.210, 1.441, −1.395). These gate units by noise level.
- **W3: what the second-layer units draw.**
  - Unit 13 draws a white left edge from (1,5) to (8,1): p17 .129, p40 .328, p51 .282, p63 .368, p74 .450, p86 .457, p97 .351.
  - Unit 21 draws the mirror-image right edge from (1,6) to (8,10): p18 .350, p31 .299, p43 .464, p56 .387, p68 .318, p81 .419, p106 .301.
  - Unit 11: row 8 white (p101 .273), row 9 black (p110–116: −.24 to −.42).
  - Unit 12: row 6 white (p77 .295), row 8 black (p101 −.328).
  - Unit 20: row 10 white (p122 .403), and row 9 too (p110 .168).
  - Unit 9: row 10 black (p121 −.442).
  - Unit 35: a solid lower interior (p99 .309, p100 .343, p90 .261).
  - Unit 29: a thick upper-left wedge (p28 .517, p39 .500, p51 .440).
  - Unit 37: top centre 2–3 px wide (p30 .183, p41 .182, p52 .181), black just to its right (p31 −.494).
  - Unit 23: black right edge (p42 −.356, p80 −.346), white interior left of it (p77 .139).
  - Units 16, 22, 30, 31, 60, 63: nearly uniform negative weights over all pixels (global darkening).

**Reasoning**

- **A white shape on black.** It never touches row 0 or columns 0 and 11.
- **The shape points up.** The envelope is narrow at top centre and full width from about row 5. The edges are straight, paired diagonals with the same slope (about 4 columns over 7 rows), and separate "white row / black row below" units show a flat horizontal bottom edge. That is an upward-pointing triangle.
- **Other shapes are ruled out.**
  - Squares and circles: the top corners are never active.
  - Letter A: there are no crossbar or hole features, and the lower interior is painted solid (unit 35).
  - Arrow or tree: the bottom detectors are full-width rows, with no narrow trunk.
  - Mountain anchored to the bottom: rows below a base at row 6 or 8 are black.
- **It is filled, not an outline.**
  - The row detectors weight the rows above their main row with the same sign; in an outline those interior pixels would be black.
  - Units 6 and 27 respond to "dark on one side, white on the other".
  - The apex detector likes (3,4), (3,5) and (3,6) together.
  - W3 paints thick wedges and solid blobs.
- **Size and position vary.** The base row ranges from 6 to 11, edges appear at several offsets, and row 1 is active across columns 3–8. So size and horizontal position vary somewhat. The sharpest pair (units 13, 21, 11) is a big centred triangle.

**Images**

Each sample is one solid white, upward-pointing, roughly isosceles triangle on a black background.
- **Apex:** at the top centre, around (1,5)–(1,6), sometimes at row 2 or shifted 1–2 columns.
- **Sides:** straight, widening by one pixel per side about every 1.75 rows.
- **Base:** flat, usually at row 8 (sometimes 9–10), spanning about columns 1–10.
- **Size:** about 8–9 px tall and 10 px wide, so it fills most of the frame. Smaller versions (base at row 6–7) and lower or larger ones also occur.

Row 0 and columns 0 and 11 always stay black.

**Example**
```
............
.....##.....
....####....
....####....
...######...
...######...
..########..
..########..
.##########.
............
............
............
```

**Confidence:** 60%