**Observations** (ln weights folded into the matrices; "×48" means the score for unit-variance inputs fully aligned with a singular vector)

- **Layer-0 layer norm.** `ln1_w.0` is bimodal: about 208 of 384 entries are below 0.15, and the largest is 1.03.
- **Layer-0 attention scores (QK) split into two groups.**
  - Four heads have almost no query–key interaction: Frobenius norms are H0 0.44, H1 0.24, H4 0.85 and H5 0.62. Each has only a rank-1 bump, with top singular value ×48 of 20, 10, 34 and 29. These heads attend nearly uniformly.
  - H2 has a QK norm of 32.6 and about 20 large singular values (543, 490, 467 … ×48). Its query and key spaces are the same subspace (overlap 0.95–0.99).
  - H2's eigenvalues come in rotation pairs (angles 52°, 104°, 69°, 157°, …) plus a large real negative one (−313). This is the signature of attention by relative position with offset −1 (the "previous" head).
  - H3 is rank 2, symmetric and positive (138, 46), sitting on a positional plane in which H2 rotates by 44°. That makes it local attention.
- **What the layer-0 heads write (OV).**
  - H2's output is high-rank and flat (0.79, 0.72, 0.68 … over about 20 dimensions), which is consistent with copying the token. Downstream reads it strongly:
    - MLP0 reads it at 3.3× (other heads about 1.7×).
    - Layer-1 values read it at 0.55–0.77 (the broad heads' outputs get 0.06–0.14).
    - The MLP0 and layer-1 readers show a knee at about 19–22 dimensions.
  - The broad heads' outputs are rank 2–3 (8.07, 3.33, 2.1, then 0.67) and read the positional plane. All layer-1 keys read this output at 17–20× baseline, layer-2 keys at 4–8×.
  - In layer 1 the query–key block on these outputs is a 2-D rotation (eigenvalues about 34 ± 100j).
- **An induction circuit in layer 3.**
  - L3H1's query reading the current token × key reading H2's output is enriched 2.5×, with eigen-score +0.96. The composite eigenvalues are positive (19.4, 16.2, 13.7 …).
  - Its OV (norm 40.9) puts 47% of its energy into the output subspace of `W_proj.3`, against a 7.5% baseline.
- **Last-layer heads.** L3H5 has QK 6.7 but OV 45.4, so it is broad. Across L1–L3, OV eigenvalues are mostly negative (−0.8 to −0.95).
- **MLP0.** Only 374 of 1536 neurons are alive (row norm above 1).
- **Output layer.** `W_proj.3` has about 28 dense, Gaussian-loaded directions (26.3 … 11.6, then 10.4, 8.35, 8.06, 7.91, 7.46, 7.17), then 5.83 and 4.07, then a floor of about 3. The copy head writes into all of them, including the weaker ones. This suggests all 28 characters are real prediction targets.

**Heads**
- L0H0: broad — near-zero QK (0.44); its rank-2 OV averages a positional plane.
- L0H1: broad — QK 0.24, same output direction as H0/H4 (cos 0.96).
- L0H2: previous — high-rank relative-position QK with the −313 parity eigenvalue; copies about 20 token dimensions.
- L0H3: local — rank-2 symmetric positional QK in the 8-character-period plane.
- L0H4: broad — QK 0.85, output cos 0.99 with H1.
- L0H5: broad — QK 0.62, low-rank positional OV.
- L1H0: local — keys from broad-head and slow-position features, 2-D rotation; values read the token and previous-token copy.
- L1H1: local — same pattern.
- L1H2: local — same pattern.
- L1H3: local — same pattern.
- L1H4: local — same pattern.
- L1H5: local — same pattern.
- L2H0: local — keys read the broad-head features at 6.6×, positional rotation, flatter content QK.
- L2H1: unclear — weaker version of the same pattern.
- L2H2: local — same pattern.
- L2H3: local — same pattern.
- L2H4: local — same pattern.
- L2H5: local — same pattern.
- L3H0: unclear — symmetric positive "similar-context" matching (trace +27; token×token block +0.33).
- L3H1: copy — current-token query × previous-token key, eigen-score +0.96; OV writes into the output space.
- L3H2: local — keys read the broad-head features (4×).
- L3H3: local — keys read the broad-head features (4.6×).
- L3H4: local — keys read the broad-head features (2.9×).
- L3H5: broad — tiny QK (6.7) with a huge OV (45.4).

**Reasoning**
- **Order.** The text is almost certainly as is. There is a sharp previous-character head copying the token, MLP0 builds bigram features from it, and a genuine induction head matches the current character against the previous-character slot. None of this helps on shuffled lines, where order carries no information. Bag-of-characters heads, which shuffled text would need, are absent: the broad heads average positions, not characters.
- **Against romaji.** About 28 actively trained output directions means all 28 characters are used. Hepburn romaji never uses l, q, v or x, so it would show only about 23.
- **English over Python.**
  - Copying is modest: one clear induction head, and only in the last layer.
  - Layers 1–2 are uniform local/positional aggregators.
  - Python, with its constantly repeated identifiers, should produce stronger and earlier copy heads.
  - The output spectrum (22 strong directions plus about 5 weak ones) fits English's few rare characters (j, x, q, z, newline).
- **Python stays possible.** The broad, newline-like position features keyed by many heads could reflect line structure.

corpus: english
order: as is
probabilities: python=0.33 romaji=0.15 english=0.52; as is=0.88 shuffled=0.12