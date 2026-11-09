**Observations**

- **Layer-0 subspaces.** The layer-0 read directions split into two subspaces with almost no overlap (overlap about 0.05):
  - Subspace A is read by the QK of L0H3 and L0H0 and by the values of L0H1/2/4/5. I take it to be position.
  - Subspace B is read by the values of L0H3 and L0H0. I take it to be the character identity.
- **L0H3 is a sharp positional head.** Its QK is high-rank (effective rank 29; D-scaled singular values 1.95, 1.79, 1.61 … about 0.3 at the 29th). Its eigenvalues come in complex pairs at about ±40–50° (1.53@±44°, 1.13@±39° …), and it reads only A. Its OV reads B (0.91 of its norm) with effective rank about 20–30.
- **MLP0 builds on L0H3.** MLP0 reads B (0.263) and L0H3's output O3 (0.236), against a baseline of 0.052. So it combines the current character with the one moved by L0H3.
- **A 2-D position plane.** L0H1, H2 and H4 have rank-1 QK (top 1.20, 0.82, 0.64). Their keys prefer one token: the component that also dominates MLP0's top direction (profile 0.64), presumably space. They partly prefer two other tokens. L0H5 has an almost zero QK (0.26) and a huge OV (16.3, 8.8). All four write into one 2-D plane P, with output overlaps 0.83–0.98.
- **The plane drives later keys.** All layer-1 heads put about 40% of their QK norm on the key reading P, through a rotation-like 2×2 block (eigenvalues about 0.6±1.8i). This pair is the ±63–82° eigenvalue pair seen in nearly every later head. The first layer-1 bias (ln1_b.1) lies mostly in P (2.6 of 3.0).
- **Layer 1 is sharp and avoids repeats.** Layer-1 QK is large (top singular value about 24). The current-token versus current-token block is negative in all six heads (trace-score −0.4 to −0.5, eigenvalue real part −0.8), so they avoid the same character or self. Their values read O3 (0.20–0.29) and B (0.11–0.18).
- **An induction circuit.** For L3H3 the matching matrix Bᵀ·QK(L3H3)·OV(L0H3)·B has trace-score +0.64, 12 sd above random, and 92% of its eigenvalue mass is real-positive. Its OV (norm 37.5) reads B and writes 35% of its norm into the output space. No other head has a comparable score.
- **Composition.** K-composition of L0H3 into layers 1–2 is at baseline (0.05), while V-composition is 0.11–0.15.
- **Vocabulary size.** After removing position and layer-0 attention-output directions, MLP0's character subspace has a knee at about 25–27 dimensions. ICA finds about 27 sparse character-detector components. W_proj.3 has about 28 directions above its bulk: 24 strong (≥8.2), then 6.4, 5.8, 4.6, 4.6, then 3.8 and below. The 25th of these is sparse, using specific neurons.
- **Which characters L0H3 carries.** It carries most characters strongly (gain 0.4–0.6). Four it barely carries (0.11–0.17): the space-like token, a second boundary-like token, and two others.

**Heads**

- **L0H0:** local. Positional QK with two positive-real eigenvalues (0.63, 0.39), and OV moves character identity (B, rank about 20). It averages the last few characters.
- **L0H1:** local. Rank-1 QK: the query reads position, the key prefers the space-like token. It writes that token's position into plane P (OV 3.4, 2.0), i.e. where the current word started.
- **L0H2:** local. Same as L0H1 (QK 0.82, OV 2.6/1.4).
- **L0H3:** previous. High-rank purely positional QK, OV carries the character (B to O3, 0.91). Its output feeds MLP0, the layer-1 values, and the keys of the induction head.
- **L0H4:** local. Same boundary-tracking pattern as L0H1/H2 (QK 0.64).
- **L0H5:** broad. QK is almost zero, so attention is near-uniform. The huge low-rank OV (16, 9) averages positional features into P.
- **L1H0:** local. Sharp QK; keys read P (0.39); avoids the same character; moves the characters (B and O3) just behind the query.
- **L1H1:** local. Same pattern (P 0.42).
- **L1H2:** local. Same pattern (P 0.45).
- **L1H3:** local. Same pattern (P 0.45).
- **L1H4:** local. Same pattern (P 0.42); also avoids the query's previous character.
- **L1H5:** local. Same pattern (P 0.41).
- **L2H0:** local. Keys still read P (0.22); high-rank content QK; no matching.
- **L2H1:** unclear. High-rank content QK (effective rank 49); little positional or matching structure.
- **L2H2:** local. Some P reading (0.11); avoids keys with the same previous character.
- **L2H3:** unclear. High-rank content QK, nothing distinctive.
- **L2H4:** local. P 0.15; weak induction-like signal (+0.17).
- **L2H5:** unclear. High-rank content QK.
- **L3H0:** unclear. Similarity-matching QK (eigenvalue real part +0.45), possibly fuzzy copying.
- **L3H1:** unclear. Weaker QK, large OV (26.5).
- **L3H2:** unclear.
- **L3H3:** copy. Strongest induction match (+0.64/+0.92): the query's current character matches the key's previous character, carried by L0H3. Its OV reads the attended character and writes into the output space.
- **L3H4:** broad. Weak QK (|QK| 5 versus 10–16 for the other layer-3 heads) with the largest OV in the model (49.2), writing into the output space. It acts as a context average or cache.
- **L3H5:** unclear. Fuzzy matching (eigenvalue real part +0.38).

**Reasoning**

- **Order.** The network relies on character order in three ways:
  - a sharp previous-character head whose output MLP0 combines with the current character;
  - layer-1 heads that gather the specific characters, and their predecessors, just behind the query;
  - a clean induction circuit (L3H3 keys on the character moved by L0H3, then copies).

  In line-shuffled text, neighbouring characters and "what followed an earlier occurrence" carry essentially no information beyond a bag of letters. So this points to text **as is**. A shuffled model would be dominated by broad, line-wide averaging heads instead.
- **Against romaji.** The character subspace (about 25–27 dimensions) and the output space (about 28 directions, sparse up to the 25th) point to essentially all 28 characters being used. Romaji never uses l, q, x and hardly v, so it would give about 23–24.
- **Against Python.** The context machinery is organised around one boundary token, very likely space: three layer-0 heads plus a broad head build a "position of the last boundary" plane that every layer-1 head keys on. That looks like word structure. Newline appears only as a weaker boundary lumped with space, and there is no dedicated line-start or indentation tracker. A Python model would need one to count leading spaces. Copying is also modest: one induction head, only in the last layer, where code would favour stronger and earlier copying.

Taken together, English is most likely. The corpus call rests on unlabelled characters, so it is much less certain than the order call.

corpus: english
order: as is
probabilities: python=0.27 romaji=0.20 english=0.53; as is=0.88 shuffled=0.12