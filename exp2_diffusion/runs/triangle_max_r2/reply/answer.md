**Observations**

- **The frame is never used.** All W1 rows for row 0, column 0 and column 11 are tiny (|w| ≤ 0.03; e.g. the max for pixel 0 is 0.018 and for pixel 143 is 0.016). Active pixels reach 0.4–0.63. The same is true for (1,0–2), (1,9–11), (2,0–1), (2,10–11), (3,0–1) and (3,11). On these pixels b3 is −0.33 to −0.54. Every high-bias h2 unit (b2 ≈ 0.37–0.55: units 4, 16, 22, 26, 30, 31, 49, 58, 60, 63) also adds a flat negative offset of −0.03 to −0.12 there.
- **The pixels the network reads form a peak.**
  - Row 1: only cols 3–8, and weakly (max 0.389, pixel 17).
  - Rows 2–3: cols 2–9.
  - Rows 4–11: cols 1–10.
  - So the region is narrow at the top and full width at the bottom. Row 11 is read too: h1 unit 38 has 0.13–0.33 on pixels 133–142.
- **W3 rows are straight-side templates.**
  - h2 13 (+) traces a "/": (1,5) .129, (2,5) .136, (3,4) .328, (4,3) .282, (5,3) .368, (6,2) .450, (7,2) .457, (8,1) .351.
  - h2 21 (+) is its mirror "\": (1,6) .350, (2,7) .299, (3,7) .464, (4,8) .387, (5,8) .318, (6,9) .419, (7,10) .228, (8,10) .301.
  - Together these form a Λ with its apex at (1,5)–(1,6).
  - Other sides sit at other offsets and slopes:
    - h2 8: (5,8) .436.
    - h2 34: (3,8) .556, (5,9) .619, (6,10) .393.
    - h2 15: (8,8) .422, (9,9) .686.
    - h2 3 (negative "/"): (3,6) −.360, (5,5) −.459, (6,4) −.427.
    - h2 17 shifts the right side by one pixel: (7,7) +.373, (7,8) −.371, (8,9) −.441.
- **Horizontal bottoms.**
  - h2 12: + on row 6 (.217, .211, .295), with rows 7–9 negative (row 8: −.21 to −.33).
  - h2 38: + on row 7 (.306, .405, .465, .394).
  - h2 11: + on row 8 (.273) and − on row 9 (−.36 to −.42).
  - h2 36: + on row 9 (.455, .371).
  - h2 20: + on row 10 (.403 … .353).
  - h2 43: + on row 10, cols 2–7 (.31–.49).
- **No vertical-edge or curved templates exist anywhere.**
- **Interior evidence.**
  - h1 unit 0's input weights form a solid triangle. Apex: (2,5) .305, (3,5) .376. Interior: (5,5) .107, (6,6) .091, (7,6) .112. Row 8: .194 .474 .299 .191 .170 .177 .246 .281 .201. Row 10 is slightly negative.
  - h1 unit 41 has a "\" band plus positive interior weights: (4,6) .177, (6,7) .200.
  - Several W3 rows paint solid blocks:
    - h2 1: rows 2–4, cols 5–8 (.198, .174, .169), with row 6 at −.35 to −.39.
    - h2 29: an upper-left wedge, (2,4) .517, (3,3) .500, (4,3) .440.
    - h2 35: rows 6–9 (.309, .343, .261, .281).
    - h2 42: rows 7–8, cols 2–5 (.21–.31).
- **b3 shape.** In row 1, b3 is least negative only at (1,5) −.143 and (1,6) −.105. Below that, the near-zero zone widens: row 2 cols 5–7, then rows 5–9 cols 2–9.
- **Time inputs.** Only the slow frequencies (inputs 148–151 and 156–159) have large weights, up to ±1.4.
- **Dead units.** h2 units 6, 24, 32, 41, 55, 57 and 59 have unstructured ±0.1 W3 rows (noise even on the frame) and b2 ≤ 0.02, so they are never active.

**Reasoning**

- The "/" and "\" sides converge to one apex at the top centre and diverge downward. Horizontal bottom lines at rows 6–10 (and row 11) close them. This is an upward-pointing, roughly isosceles triangle with a flat base, symmetric about the line between columns 5 and 6.
- Ruled out:
  - **Letter "A":** h2 12's bar darkens everything below it, including where the legs would continue.
  - **Squares:** there are no vertical edges.
  - **Circles:** there are no curved strokes.
  - **Inverted triangles:** the readable area and b3 are narrow at the top.
- **Solid, not outlined:** the detectors put positive weight on interior pixels, and several output units paint solid patches. With outlines, interior pixels would anticorrelate with the sides.
- **Size and position vary:**
  - Sides have different slopes, e.g. 4 columns per 7 rows (h2 13) versus 3 per 5 (h2 34).
  - Apexes appear at (1,4) through (1,7), (2,6) and (3,6).
  - Bases appear at rows 6–11.
- A 1-pixel black margin is kept at the top, left and right; the base can reach the bottom row.

**Images**

Each image is a single solid white triangle pointing up, on a black background:
- **Apex:** at the top centre, usually row 1–2 at cols 5–6, sometimes row 3 or shifted 1–2 columns.
- **Sides:** straight and staircased.
- **Base:** flat and horizontal, low in the frame (typically row 8–10, range about 6–11).
- **Width:** the base spans about 6–10 pixels, often nearly the full width (cols 1–10).
- **Size:** about 6–10 rows tall, covering roughly half the image.

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
.##########.
............
............
```

**Confidence:** 70%