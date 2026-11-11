**Observations:**
- **L0H3 is a previous-character head.**
  - Its query-key matrix is high-rank (singular values 1.95, 1.79, 1.61, 1.45 …).
  - It is dominated by rotation: antisymmetric part 4.08 versus symmetric part 3.82.
  - Its eigenvalues come in complex pairs at ±44°, ±39°, ±45°, ±51°, ±60°, ±84°. That is the signature of attention to a fixed offset back in position.
  - Its value/output map reads character directions, not its own query-key planes (only 0.17 overlap, 0.10 expected by chance). It has a flat spectrum (about 18 values from 1.07 down to 0.49) and writes into a new subspace (overlap 0.09).
  - All six L1 heads read that output in their values, at 5–10 times chance.
- **L3H3 is an induction (copy) head built on L0H3.**
  - With the query taken from the current character and the key from what L0H3 wrote, the matching matrix correlates +0.52 with the identity. Its eigenvalues are nearly real and positive (0.43@0°, 0.34@±25°, 0.30@±16°).
  - Every other head scores ≤0.13 on this test.
  - L3H3's values read the character embeddings (2× chance), so it copies the character it attends to.
- **L0H1, L0H2, L0H4 and L0H5 form a position-code machine.**
  - Their query-key matrices are close to rank 1: singular values 1.2/0.34, 0.82/0.14, 0.64/0.17 and 0.26/0.03.
  - Their value/output maps have rank about 2 and read the same positional direction (cosines 0.63–0.93). L0H5 does this with gain 16.3.
  - They write into one shared 2-D code: their first output directions agree at |cos| 0.83–0.98 (L0H1 vs L0H2: −0.98), and they share a second direction w.
  - Almost every later head reads w in its keys, at 0.85–0.90 of its spectral norm (only L3H3 and L3H4 do not). Values barely read it (≤0.05).
- **Recency strength falls with depth.**
  - The fixed query bias (from the LayerNorm bias) times the position code gives score sensitivities of 7.5–12.9 in L1, 4.0–10.1 in L2 and 1.0–2.1 in L3.
  - The largest query-key singular value follows the same pattern: about 9 in L1, 4–6.5 in L2, 1.9–3.6 in L3.
- **The output side suggests all 28 characters are predicted.**
  - W_proj.3 has 21 large singular values (28.5 down to 11.05), then 9.08, 8.35, 8.24, then 6.37, 5.84, then 4.63, 4.58, then a smooth tail from 3.78.
  - The copy head L3H3 writes above chance into directions 0–27 except one: 2.3× on 0–20, 1.4–2.0× on 21–27 (direction 24 at 1.0×), and 0.9× from 28 onward.
  - That gives about 27 logit dimensions, i.e. 28 characters minus one. All 28 characters seem to be predicted, including copying of rare letters.
  - These output-side counts are indirect inferences from weight spectra, not a direct readout of the vocabulary.

**Heads:**
- L0H0: local — top query-key component is positional on both sides (0.74/0.64 in positional planes); real positive eigenvalues (0.63@0°, 0.39@0°), so it attends near its own position.
- L0H1: broad — rank-1 query-key with a character-type key and a positional query; it attends to every occurrence of one character type and copies their position into the shared code.
- L0H2: broad — same mechanism as L0H1 (0.82 vs 0.14; output cos −0.98 with L0H1).
- L0H3: previous — rotational positional query-key (±35–84°) and a value/output map that copies character identity; read by L1 values and by L3H3's keys.
- L0H4: broad — rank-1, the same family as L0H1 and L0H2.
- L0H5: broad — query-key almost zero (0.26), so near-uniform; reads position with gain 16.3 and writes the "position clock" w.
- L1H0: local — keys dominated by the position code (largest singular value 9.5); strong recency (bias 12.9); avoids its own character (−0.44); values read the previous character.
- L1H1: local — same pattern (9.2; bias 11.9; −0.39).
- L1H2: local — same pattern (9.4; bias 10.0).
- L1H3: local — same pattern (9.5; bias 11.0).
- L1H4: local — same pattern (8.5; bias 7.5).
- L1H5: local — same pattern (9.2; bias 12.5).
- L2H0: local — keys read w at 0.89; sensitivity 5.2, bias 8.6.
- L2H1: local — 4.1 / 8.2.
- L2H2: local — 3.9 / 6.7.
- L2H3: broad — weakest recency (3.5 / 4.0); flat spectrum (largest/second singular value 1.3).
- L2H4: local — 5.7 / 10.1.
- L2H5: local — 4.5 / 9.1.
- L3H0: broad — weak recency (2.9 / 2.1); flat query-key (ratio 1.1); values read MLP2.
- L3H1: broad — 2.2 / 1.0.
- L3H2: broad — 2.6 / 1.9.
- L3H3: copy — induction via L0H3 (+0.52, versus ≤0.13 for every other head); ignores the position code; values copy the character it attends to.
- L3H4: broad — small query-key (1.9), ignores the position code (0.25); a near-uniform average of MLP2 features (value/output singular value 21.3).
- L3H5: broad — 3.0 / 2.1, flat spectrum.

**Reasoning:**
- **Order: as is.** A sharp previous-character head and an induction head built on top of it are only useful when character order inside lines is real. With shuffled lines, everything beyond the bag of characters in the line is noise, so I would expect only broad within-line heads and no previous-character/induction circuit.
- **Romaji is unlikely.** Hepburn romaji essentially never uses l, q, x or v, so it would show only about 23–24 active logit dimensions. The output side shows about 27, i.e. all 28 characters are predicted and copied.
- **English fits better than Python.**
  - The 21 strong plus 7 weaker logit dimensions match English letter frequencies: space and 20 common letters, then rarer v, k, newline, x, j, q, z.
  - Python code has a flatter letter distribution (x, k, v, y, b are common in identifiers), so I would expect about 24 strong and 4 weak.
  - Highly repetitive code would also be expected to grow several early induction heads. Here there is a single induction head, in the last layer.
  - The rest of the network fits word-level English: word and position machinery plus local character gatherers.

**corpus:** english
**order:** as is
**probabilities:** python=0.22 romaji=0.15 english=0.63; as is=0.90 shuffled=0.10