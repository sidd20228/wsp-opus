**Observations**

- **Layer 0 has two sharp relative-position heads, L0H1 and L0H5.**
  - Their W_q and W_k read the same roughly 7-dimensional subspace (principal-angle cosines 0.97 to 0.82). That subspace is unrelated to what their W_v reads (overlap at the level of random subspaces).
  - QK singular values are 4.81 for H1 and 3.28 for H5.
  - Restricted to that subspace, their QK matrices have rotation-type complex eigenvalues: 1.74±2.76i (59°) for H1 and 1.65±2.2i (53°) for H5.
  - The two QK matrices nearly commute: normalized commutator 0.07, against 0.19 for random conjugates. That is what attention by a fixed relative offset looks like when positions rotate in shared planes.
  - Their values copy the full character identity. After removing the positional directions, W_v of H1 has exactly 27 strong singular values (2.25 down to 1.00), then a gap to 0.71.
- **L0H4 is uniform.** Its W_q and W_k are essentially zero (QK singular values ≤ 0.04). Its W_v reads only a 2-dimensional positional plane (singular values 2.9 and 2.0; cosines 0.98 and 0.92 with the main H1/H5 plane). Its OV is huge (13.9, 9.7). L0H0 reads the same plane with weak QK (0.71, 0.54).
- **About 27–28 characters are actively used.**
  - The 27 token directions read by L0H1 are confirmed as character directions: the layer-0 MLP reads them (overlap 0.44 to 0.62), and the positional Q/K directions barely touch them (about 0.1 to 0.25).
  - W_proj.3 has about 28 large singular values (28.2 down to 7.9; then 6.5 and 5.3), so about 28 output symbols are distinguished.
  - Four token directions are weaker (W_v singular values 1.19 to 1.03 against a top of 2.25), consistent with a few rare letters.
- **Induction circuit.** I composed the query/key matrix of later heads with the L0H5 output, restricted to the character directions. Three heads show strong positive "match" structure: trace/Frobenius +2.75 for L2H1, +2.62 for L3H1 and +2.80 for L3H5 (eigenvalue positivity 0.87 to 0.91). L3H0 is weaker at +1.25. Every other head lies between about −0.5 and +0.7.
- **Layer 1.** All six heads have keys dominated by the L0H0/L0H4 output (61 to 72 times the per-dimension baseline), which marks the start of the window.
  - They also share one static character preference coming from the LayerNorm bias: |g| of 3 to 10, with one dominant principal component (singular value 18.4 against 5.1).
  - Their own-character query/key form is strongly negative (trace/Frobenius about −2.2), i.e. they avoid the same character.
  - Their queries barely read position (about 1× baseline), so they have no relative recency.
- **Layer-0 MLP.** About 1,000 of its 1,536 neurons are untrained or dead.

**Heads**
- L0H0: broad. Weak QK; key reads one character class; value copies the coarse positional plane.
- L0H1: previous. Sharp rotation QK (about 59°), commutes with H5; copies 27 character directions.
- L0H2: local. Moderate positional QK (1.06), main-plane angle about −17° (near itself), modulated by character in the key.
- L0H3: local. Positional QK (1.32), main-plane angle about 179° (roughly 3 back); copies position and character.
- L0H4: broad. QK ≈ 0, so uniform over the prefix; huge OV of a positional plane (marks the window start).
- L0H5: previous. Rotation QK (about 53°); it is the head the induction heads use for their keys.
- L1H0: broad. Window-start key 33×; static character preference; anti-same-character.
- L1H1: broad. Same pattern (key 69×, |g| = 10.5).
- L1H2: broad. Same pattern (key 67×, |g| = 7.6).
- L1H3: broad. Same pattern (key 72×, |g| = 7.3).
- L1H4: broad. Same pattern (key 70×, |g| = 8.3).
- L1H5: broad. Same pattern (key 61×, |g| = 8.6).
- L2H0: broad. Window-start key 8.5×, static preference 2.1, no match structure.
- L2H1: copy. Current character matched against the L0H5 output, +2.75 (positivity 0.88).
- L2H2: unclear. Window-start key 17×, but anti-match with L0H5 (−1.80) and a strong previous-to-current term; its output is what L3H5 composes with (+0.95).
- L2H3: broad. Window-start key 17×, static preference 3.4.
- L2H4: broad. Window-start key 14×, static preference 3.2.
- L2H5: broad. Window-start key 6×, weak content.
- L3H0: copy (weak). Match with L0H5 is +1.25.
- L3H1: copy. Match with L0H5 is +2.62.
- L3H2: unclear. Weak window-start key (4×), small same-character term (+0.38).
- L3H3: unclear. Small QK throughout.
- L3H4: unclear. Weak window-start key (3×), weak match (+0.70).
- L3H5: copy. Match with L0H5 is +2.80; also matches the same character and the same previous character.

**Reasoning**

- **Order.** Sharp offset-1 heads that copy the whole character, plus composed induction heads matching the current character against the previous-character channel, only pay off when character order within a line is real. In shuffled lines, the previous character is just one random sample from the line's letters, and induction is useless; one would expect broad within-line averaging instead. So the text was used as is.
- **Corpus: romaji is unlikely.** About 27 well-learned character directions on the input side and about 28 output directions mean almost all 28 symbols are in regular use. Hepburn romaji uses only about 22 letters plus space and newline (l, q, x and v are essentially absent).
- **Corpus: English vs Python is only a lean.**
  - For English: there is no long-range relative-position machinery of the kind needed to count Python indentation or track columns. All position-sensitive heads in layer 0 are local (offsets up to about 3), and later heads use only a window-start signal.
  - For English: there is a single dominant structural attention target, which fits space as the only important boundary character (newlines are rare in Wikipedia paragraphs).
  - For English: a small group of about 4–5 weaker character directions fits English's very rare letters (x, j, q, z, and newline).
  - For Python: the induction and same-context heads (L3H5) would also fit code's repetitiveness. So English is only moderately favoured.

corpus: english
order: as is
probabilities: python=0.37 romaji=0.10 english=0.53; as is=0.93 shuffled=0.07