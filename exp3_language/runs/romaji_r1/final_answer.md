**Observations**

All numbers come from the weights alone. QK and OV were scaled by the ln1 weights.

- **L0H3 is a fixed-offset positional head.**
  - It has the largest layer-0 QK (|q|=14.1, |k|=14.7) with a flat spectrum (1.95, 1.79, 1.61, …).
  - 96% of its QK lies in one ~20-dimensional subspace S. Its Q and K read the same space (principal cosines 0.99 … 0.85).
  - Restricted to S, its eigenvalues are rotations: 1.56@44.6°, 1.16@38.9°, 1.11@45.8°, 1.08@51.1°, 1.06@61°. The trace is +10.8.
  - The MLP0 does not read S (overlap 0.18, against a chance level of 0.16), so S is the position-embedding space.
  - Its value side reads tokens and copies about 18–20 dimensions (OV singular values 1.13 … 0.53, then a drop).
- **MLP0 does bigram processing.** Its read space is made of current-token directions plus L0H3's output: L0H3's output overlaps MLP0's read space by 0.77, and each of the top 40 MLP0 read directions is 15–56% L0H3 output.
- **There are exactly 26 token directions.** After removing the attention-output directions, MLP0's read spectrum is 13.5 … 5.01, 4.82, then drops to 3.88 (gap ratio 1.24–1.31, stable however many output directions are removed). Those 26 directions overlap the token-reading values of L0H0/L0H3 by 0.21–0.59 each. Every later direction is at chance level (0.03–0.07).
- **L0H1, H2 and H4 are near-duplicates, and with L0H5 they build a sink.**
  - In H1/H2/H4 the query reads an absolute-position direction (cosine 0.89–0.91 across the three heads). The key reads a special direction that only layer-0 keys use: read strength 8.7, 5.9 and 4.1 for H1, H2, H4, against about 1 for MLP0 and all later layers.
  - Their values read position (0.60–0.72), and their outputs are rank 2 (output cosines 0.83–0.98).
  - L0H5 has almost no QK (top singular value 0.26), so it attends nearly uniformly. Its OV is huge (16.3, 8.8), rank 2, and reads position.
  - These outputs are read only by later keys. In every L1/L2 head the query-bias key term W_kᵀW_q·ln1_b has norm 19–32 in L1 and 8.5–18.5 in L2. 80–86% of it lies in this 4-dimensional space, always pointing the same way (about [+1, −0.8]).
  - This is an attention sink at the start of the window, not tracking of any token.
- **L1 and L2 heads are local and content-based.**
  - L1 heads are sharp (top QK singular values 21–25).
  - Their token×token QK has a strongly negative trace (trace/norm −1.5 to −2.2), so they avoid the current character and favour recent different ones.
  - L1 and L2 heads share one positional rotation plane, with phases 62–79° (L0H3's are 41–60°).
- **There is one clear induction circuit.** For L3H3, the query's current token matched against the key's L0H3 output gives trace/norm +2.72 (random spread about 0.21), with 0.94 of the eigenvalue mass positive.
  - With other token bases it is still +2.5.
  - No other head exceeds +0.72 (L2H4).
  - L3H3 also carries a recency term (positional eigenvalues ≈ +0.27 at 5–8°) and ignores the sink.

**Heads**

- L0H0: local — 83% of its QK is positional, with real positive eigenvalues (0.81@0°, 0.46@0°), i.e. a recency window; the value reads tokens.
- L0H1: broad — weak QK (top singular value 1.2); query on absolute position, key on a special positional direction; copies position; feeds the sink.
- L0H2: broad — duplicate of L0H1 (output cosine 0.98).
- L0H3: previous — rotation-structured positional QK (phases 34–61°), copies about 20 token dimensions, heavily read by MLP0 and by L3H3's keys.
- L0H4: broad — third duplicate of L0H1.
- L0H5: broad — almost no QK (uniform attention), huge rank-2 positional OV that drives the sink.
- L1H0: local — sharp; avoids its own token (trace/norm −2.2); positional plane at 75°; sink.
- L1H1: local — same pattern; mildly prefers keys equal to the previous character (+0.51).
- L1H2: local — same pattern; plane at 77°.
- L1H3: local — same pattern; plane at 62°.
- L1H4: local — avoids both the current and the previous character (−1.66 / −1.95), so it looks a few characters back.
- L1H5: local — same pattern as L1H1; plane at 79°.
- L2H0: local — moderate QK (top singular value 9.8); shared plane at 67°; sink; avoids its own token.
- L2H1: local — same pattern; plane at 76°.
- L2H2: local — same pattern; plane at 67°.
- L2H3: unclear — weakest positional part and smallest sink term in L2.
- L2H4: local — plane at 70°; weak induction signal (+0.72 or +0.4 depending on basis).
- L2H5: local — plane at 69°.
- L3H0: local — real positive positional eigenvalue (0.29@0°) plus a rotation; content-based.
- L3H1: broad — small, flat QK (top singular value 5.3) and large OV.
- L3H2: local — positional eigenvalue 0.19@0° plus a rotation at 72°.
- L3H3: copy — matches the current token against the key's previous token (+2.72); recency bias; no sink.
- L3H4: broad — smallest QK (top singular value 3.7) and the largest OV (29.4, about 8 strong dimensions); no sink.
- L3H5: local — positional trace +0.81, rotation at 52°.

**Reasoning**

- **Order.** A sharp offset-1 positional head that copies about 20 token dimensions, half of MLP0's input given to the previous character, and a strong induction head are all useless when the characters of each line are shuffled. There a bag-of-line (broad) strategy would dominate. So the text was used as is.
- **Corpus, against romaji.** Two independent readers agree that 26 token directions are learned. Romaji (Hepburn) uses about 22 letters plus space and newline, about 24 tokens, while English and Python use all 28. This fits English or Python better than romaji. It is not decisive: only about 22–24 of the 26 directions are used strongly.
- **Corpus, against Python.** Python's heavy repetition of identifiers and its indentation should produce several strong copy heads and line-start trackers, for example heads that key on newline and copy its position. Here there is no token-keyed position copying at all: the layer-0 position-copy heads key on a positional direction, and no later head copies position. There is only one induction head, in the last layer, with modest QK. This fits English Wikipedia: long lines, spelling handled by local heads, and one late induction head for repeated names.
- The evidence for English over Python is weaker than the evidence for the order.

corpus: english
order: as is
probabilities: python=0.28 romaji=0.20 english=0.52; as is=0.92 shuffled=0.08