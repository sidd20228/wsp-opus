**Observations:**
- **Layer 0 has one strongly positional head, L0H3.**
  - Its W_q and W_k read the same ~26-dimensional subspace (principal cosines 0.99 for 24 directions, then 0.76, 0.76, 0.46). This subspace contains no current-token directions (overlap ≈ 0).
  - Its QK eigenvalues are complex with a consistent rotation: 1.53@44°, 1.13@39°, 1.09@46°, 1.07@51°, 1.05@61°…
  - Its W_v reads character identity: the subspace it shares with MLP0's W_fc has cosines 0.93 down to 0.53 over about 22 dimensions.
  - Its OV has about 18 strong singular values (1.13 … 0.53), and MLP0 reads that output strongly (W_fc.0·OV₀₃ singular values 9.6, 6.4, 6.2 … 3.5 over about 15 dimensions).
  - So it is a sharp relative-position head that moves a nearby character into MLP0: a previous-character head.
- **Other layer-0 heads:**
  - L0H0 is a soft positional head. Its eigenvalues are real (0.63@0°, 0.39@0°), and in the shared positional plane it rotates 32°, against 45° for L0H3. Its W_v also reads tokens.
  - L0H5 is almost uniform (QK singular values 0.26, 0.03…) but has a huge, nearly rank-2 OV (16.3, 8.8, 2.8). It reads positional directions, not characters.
  - L0H1, L0H2 and L0H4 have rank-1 QKs (1.20, 0.82, 0.64). Their queries read position, their keys read position plus a shared character class (key-feature cosines 0.5–0.7), and their values read position. They report where the last character of some class occurred.
  - L0H1, H2, H4 and H5 all write the same 2-D subspace (principal cosines ≈ 1.0).
- **Most layer-1–3 heads share an attention sink.** Sixteen of them share one key-only preferred direction (cosines 0.6–0.88). It comes from the LN-bias term, with |q_bᵀk| up to 31.8, and lies about 0.9 inside that 2-D output. The six layer-1 heads also avoid matching their own character (same-character score ratio −0.7 to −0.85). Weak positional rotations of about 70° appear in the same plane.
- **There is one clear induction circuit, L3H3.**
  - Its QK is positive-definite: ratio +0.98, about 46–50 dimensions at small angles.
  - Composing its QK with L0H3's OV gives score ratio +0.89 (magnitude 3.79). No other head exceeds +0.26.
  - Its OV writes the output-reading directions: 29% of its energy lies in the top 20 output directions of the last MLP, against about 8% for the other heads.
  - The induction circuit, plus a sharp previous-character head feeding MLP0, only pays off if the character order is real.
- **The alphabet size points to 28 symbols with a long tail.**
  - With position and layer-0 head outputs removed, MLP0's reading of the current character has a stable knee at 26 dimensions: …4.91, 4.69 | 3.75. This held across all the projections I tried.
  - The last MLP's W_proj.3 singular values fall into tiers: 21 strong (28.5 … 11.05), then 9.08, 8.35, 8.24 | 6.37, 5.84 | 4.63, 4.58, then a smooth bulk from 3.78. That is 28 output directions in a 21/3/2/2 pattern.

**Heads** (labels for L0H0/H3, L0H5, L3H3 and L3H4 are fairly firm; the layer-1–2 labels are the weakest):
- L0H0: local — soft positional head (rotation 32° vs L0H3's 45°, real eigenvalues 0.63/0.39) with a token-moving value.
- L0H1: local — rank-1 QK: position-driven query, key on a character class; reports the position of the last such character.
- L0H2: local — same mechanism as L0H1 (key-feature cosine 0.7 with it).
- L0H3: previous — pure positional QK in about 26 dimensions (cosines 0.99), consistent ~45° rotation; moves token identity into MLP0; it is the head the induction head keys on.
- L0H4: local — same pattern as L0H1 and L0H2, weaker (QK singular value 0.64).
- L0H5: broad — QK almost zero (0.26), so nearly uniform; large rank-2 OV writing a position summary that later heads use as a sink key.
- L1H0: local — shared sink key (cosine 0.83–0.87), weak ~73° positional term, anti-same-character content.
- L1H1: local — same pattern (positional eigenvalue 0.84@69°).
- L1H2: local — same pattern (0.73@70°).
- L1H3: local — same pattern (0.92@62°).
- L1H4: unclear — same family, weakest positional term (0.53@57°).
- L1H5: local — same pattern (1.00@77°).
- L2H0: local — sink key plus 0.77@69° positional term.
- L2H1: local — sink plus 0.74@74°.
- L2H2: local — sink plus 0.68@70°.
- L2H3: unclear — weak QK in every component (0.28@58°).
- L2H4: local — 0.90@73° positional term, plus a slight induction-like composition with L0H3 (+0.24).
- L2H5: local — sink plus 0.83@72°.
- L3H0: unclear — positive content matching (ratio +0.72) but no composition with the previous-character head.
- L3H1: unclear — weak, mixed QK.
- L3H2: unclear — weak, mixed QK.
- L3H3: copy — induction: keys on L0H3's output (+0.89), copies into the output-reading directions.
- L3H4: broad — weak QK (|A| = 5.0, nearly uniform) with a huge OV (21, 18, 17 …): a context-frequency prior.
- L3H5: unclear — positive matching (ratio +0.73), no clear source.

**Reasoning:**
- **Order.** In shuffled lines, previous-character and induction heads are useless, and the attention would be almost all bag-of-line. Here there is a sharp relative-position head moving characters into MLP0, and an induction head built on it that copies into the output-reading directions. So the text was used as is.
- **Corpus.** MLP0 distinguishes about 26 current-character dimensions, and the output side has about 28 directions. Romaji in Hepburn style uses only about 24 symbols (no l, q, x; v is almost absent), so it is unlikely.
- **English versus Python.** The output tiers 21/3/2/2 match English letter frequencies well:
  - 21 symbols at 1.2% or more: space plus e t a o i n s r h l d c u m f p g w y b;
  - then v, k and newline;
  - then x and j;
  - then q and z.
  
  Python spreads letters more evenly and uses newline heavily, so I would expect more strong dimensions. Its heavy repetition should also produce several strong, early induction heads, but there is only one, in layer 3.

**corpus:** english
**order:** as is
**probabilities:** python=0.22 romaji=0.13 english=0.65; as is=0.87 shuffled=0.13