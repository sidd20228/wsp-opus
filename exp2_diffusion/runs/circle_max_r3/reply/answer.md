**Observations:**
- **Black frame.** The W1 rows for the outer frame pixels (row 0, row 11, column 0, column 11; for example rows 0–13 and 132–143) are all |w| ≲ 0.05. Interior pixels get weights up to about 0.45 (W1[17,9]=0.437, W1[62,0]=0.446, W1[99,16]=0.448). b3 on the frame is −0.28 to −0.61. Content is confined to the 10×10 box of rows 1–10, columns 1–10.
- **b3 has a round footprint.** The middles of the box's sides are near 0: (1,6) −0.036, (6,1) −0.064, (6,10) −0.063, (10,7) −0.032. The box's corners stay dark: (1,1) −0.483, (1,10) −0.397, (10,1) −0.470, (10,10) −0.358. The W1 rows for those corner pixels (13, 22, 121, 130) are near zero. The interior is about 0, with the maximum at the centre ((5,6) +0.182).
- **Time gating.** The slow sin/cos rows carry the large weights. For example, row 157 (cos t·f5) has 1.335, 1.401, 1.438, 1.456, −1.591, −1.290, so units switch on and off with the noise level.
  - About 10 h2 units (7, 10, 11, 22, 33, 34, 45, 58, 61, 62) have W3 rows that are negative everywhere (−0.05 to −0.25). These are background-darkening units.
  - W3 rows 13, 32, 55 and 59 are unstructured (dead units).
- **Most other W3 rows paint thin, rounded arcs one pixel wide:**
  - Unit 42 (upper-left arc): (1,2–4) 0.351 0.552 0.382, (2,1) 0.348, (2,2) 0.577, (3,1) 0.539, (4,1) 0.361. The corner pixel (1,1) is only 0.049 and the inside pixel (3,3) is −0.096.
  - Unit 41 (bottom arc): row 10, columns 3–7: 0.244 0.419 0.525 0.582 0.367. Also (9,2) 0.232, (9,7) 0.382, (8,8) 0.327. The inside (8,4–6) is about −0.03.
  - Unit 29 (top arc): row 1, columns 4–8: 0.194 0.408 0.507 0.503 0.323.
  - Unit 19 (right arc): column 10, rows 4–7: 0.360 0.519 0.519 0.344.
  - Unit 60 (upper-right arc): (2,6–8) 0.276 0.441 0.521, (3,9) 0.578, (4,10) 0.310.
  - Unit 1 (rounded lower-left): column 1, rows 6–8: 0.352 0.538 0.515, then (9,2) 0.389, with the corner (9,1) only 0.214.
  - Unit 35 (complete ring): row 3, columns 5–8: 0.168 0.211 0.237 0.159; (5,3) 0.119, (5,9) 0.243, (6,9) 0.288; the bottom (rows 8–9) 0.10–0.21. The centre (5,5–7) is −0.044 −0.020 −0.002, so the ring is hollow.
  - Unit 40 (tiny circle): a small blob at (5,5) 0.228 and (6,4) 0.210, surrounded by negatives: (4,6) −0.254, (5,7) −0.240, (6,2) −0.241.
  - Negative arcs:
    - Unit 0: column 7, rows 4–7, −0.26 to −0.40, curving to (8,6) −0.391 and (8,5) −0.311.
    - Unit 16: column 9, rows 4–6, −0.29 to −0.35, then (7,8) −0.370 and (8,7) −0.385.
- **W1 detector, unit 4.** It has negative weights on the centre: pixels 54 −0.352, 65 −0.363, 77 −0.357, 89 −0.276. It has positive weights on the right rim: 81 0.156, 82 0.174, 94 0.144. Its W3 row draws a top arc at row 2, columns 3–6: 0.295 0.467 0.500 0.393. In other words, "dark middle plus bright rim → more rim".
- **What is absent.** No unit draws straight edges meeting at square corners, diagonals across the box, crossings, or digit-like strokes. Units 15 and 24 show coarse blocks (upper-right and lower-left, 0.1–0.26), which look like "where is the object" guesses at high noise.

**Reasoning:**
- **Frame and box.** The frame is always black, and the object lies inside the 10×10 box.
- **Circles, not squares or triangles.** Shapes reach the middle of every side of the box but never its corners, and every arc has rounded corners. Squares, rectangles and triangles would light the box corners or show sharp corners.
- **Outlines, not filled discs.** The arcs are one pixel thick with neutral or negative insides. Unit 35 is a ring with an empty centre. Unit 4 looks for a dark centre with a bright rim. Filled discs would give solid interiors and edge dipoles instead.
- **Radius varies.** Some arcs run along rows 1 and 10 and columns 1 and 10 (diameter about 10). Unit 35 shows radius about 3. Unit 40 shows a tiny circle.
- **Position varies.** Arcs and blocks appear in different quadrants.
- **Not digits or letters.** There are no stroke types other than arcs.

**Images:** Each image is a single thin white circle outline, about 1 pixel wide, on a black background.
- The radius ranges from about 1–1.5 px up to about 4.5 px (diameter roughly 3–10 pixels).
- The centre position varies, but the whole circle always stays inside rows 1–10 and columns 1–10.
- The outer one-pixel frame is always black, and nothing else appears in the image.

**Example:** a radius-3 ring centred near (6,6).
```
............
............
............
.....###....
....#...#...
...#.....#..
...#.....#..
...#.....#..
....#...#...
.....###....
............
............
```

**Confidence:** 65%