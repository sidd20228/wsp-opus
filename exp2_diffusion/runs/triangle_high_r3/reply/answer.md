# What the diffusion model generates: filled white triangles, apex up, on black

## Observations

**Where the network looks (W1).** Every input-weight row for the outer frame is near zero (|w| < 0.03): row 0, columns 0 and 11, and row 1 outside columns 3–8. Those pixels never vary, so they are always black. Large weights appear only inside a trapezoid that is narrow at the top and wide at the bottom:
- Row 1: columns 3–8 only. Pixel 17 (r1c5) has 0.389 on unit 19; pixel 20 (r1c8) has 0.176 on unit 14.
- Row 2: columns 2–9. Pixel 29 (r2c5) has 0.629 on unit 19; pixel 27 has 0.314 on unit 4; pixel 32 has 0.396 on unit 14.
- Rows 3–8: columns 1/2 to 9/10.
- Rows 9–11: columns 1–10. Unit 38 reads row 11 (pixels 133–142, weights 0.129, 0.210, 0.279, 0.309, 0.323, 0.326, 0.320, 0.293, 0.242, 0.143). Unit 25 reads the bottom-left corner (r10c1–c3 = 0.371, 0.561, 0.407; r9c2 = 0.415). Unit 31 is negative on all of rows 9–10 (−0.2 to −0.33).

**Unit 0 is a filled triangle template.** Its positive weights form a widening wedge:
- r2c5 = 0.305
- r3c4–6 = 0.136, 0.376, 0.166
- r4c4–6 = 0.244, 0.207, 0.140
- r5c3–7 ≈ 0.1–0.2
- r6c3–9 ≈ 0.06–0.25
- r7c3–9 ≈ 0.09–0.28
- r8c1–9 = 0.194, 0.474, 0.299, 0.191, 0.170, 0.177, 0.246, 0.281, 0.201

The interior weights are positive, not just the edges.

**Edge detectors.**
- Unit 4 reads a left edge: r1c3 0.112, r2c3 0.314, r3c2–3 0.258/0.382, r4c2 0.405, r5c1–2 0.238/0.331.
- Unit 5 reads another left edge: r2c4 0.338, r4c3 0.416, r6c2 0.398.
- Unit 28 reads a right edge: r1c6–7 0.111/0.157, r2c6–7 0.291/0.351, r3c6–7 0.397/0.380, r4c6–8.
- Unit 18 reads the lower-left side: r6c2 0.313, r8c2 0.435, r9c1 0.432.

**Output strokes (W3).**
- Row 13 is a "/" stroke from r1c5 down to r8c2. Its weights include r3c4 0.328, r5c3 0.368, r6c2 0.450, r7c3 0.457 and r8c2 0.351.
- Row 21 is a "\" stroke from r1c6 down to r8c10. Its weights include r1c6 0.350, r3c7 0.464, r4c8 0.387, r6c9 0.419 and r8c10 0.301.
- Row 33 is a steeper "\" from r2c7 down to r6c10.
- Several rows are full-width horizontal bars (flat bases at different heights):
  - Row 38: row 7, columns 1–10 (0.140 … 0.465 … 0.208).
  - Row 20: row 10, columns 1–10 (0.237 to 0.403).
  - Row 42: row 10, columns 3–7 (0.31 to 0.49).
  - Row 12: row 6 positive and row 8 negative.
- Row 34 is a broad positive blob over rows 6–9 (about 0.1–0.34), which is fill rather than outline.

**Background units.** W3 rows 4, 16, 18, 26, 30, 46, 59, 61 and 63 are uniformly negative everywhere. They push the background toward black. Rows 6, 23, 39, 52 and 56 look like noise, probably dead units.

**Output biases (b3).** The brightest biases sit near the top centre: r1c5–6 = −0.14/−0.11 and r2c5 = −0.013. Biases around 0 cover the centre and lower middle. The corners are the darkest, at about −0.45 to −0.53.

## Reasoning

The footprint of varying pixels is one or two pixels wide at the top, near columns 5–6, and spans columns 1–10 at the bottom. This shape points to an upward-pointing shape:
- Squares or circles would make the top rows as wide as the middle, so they are ruled out.
- Digits or letters would put strokes in the upper corners. They also would not produce unit 0's solid wedge or the full-width flat bars.

The paired "/" and "\" edges meet at a top apex. Full-width horizontal bars at rows 6–11 act as a flat base whose height varies. Unit 0's interior weights and the W3 blob show the inside is filled, not just outlined. Units 31, 38 and 25 show that the bottom rows are sometimes lit and sometimes dark, so the size and base height vary. The edge units are offset from one another, so the apex and slopes shift by a column or two.

## Images

Each image is a single solid white triangle, pointing up, on a black background:
- **Apex:** near the top centre, at row 1–2, around columns 4–6.
- **Sides:** slant outward roughly 2 rows per column.
- **Base:** flat, somewhere between row 7 and row 11, up to about 10 pixels wide (columns 1–10).
- **Size:** varies from about 6 to 10 rows tall, with small horizontal shifts.
- **Always black:** row 0, columns 0 and 11, and everything outside the triangle.

## Example

```
............
.....##.....
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
```

## Confidence

45%