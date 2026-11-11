Observations
- **Layer 0 is lopsided.** Only one head has a large query–key product.
  - L0H2: ‖W_q‖≈31 and ‖W_k‖≈28. Every other layer-0 head is at 3–7, about the initialisation size of 3.1.
  - L0H2's query–key matrix has about 20 nearly equal singular values (11.3, 10.2, 9.7, 9.7, 9.3 …). Queries and keys read the same ~18-dimensional subspace (principal cosines 0.99 to 0.8).
  - That matrix is 72% antisymmetric. Its symmetric part has large negative eigenvalues (down to −8.8). Restricted to its subspace, its eigenvalues are complex pairs with angles of 31–180°.
  - This is the signature of attention at a fixed relative offset, not of self-attention. The negative and high-angle parts favour odd offsets, which points to offset 1.
  - Its value input lies in a different subspace from its query/key subspace (cosines ≤0.5 after the first). That fits a positional head that copies the previous character's identity.
- **Four layer-0 heads are nearly uniform.**
  - L0H0, H1, H4 and H5 have top query–key singular values of only 0.41, 0.20, 0.71 and 0.61, with no consistent key direction across the four heads.
  - Their value/output maps are nearly identical, rank about 3 (singular values 8.1/3.3/2.1 and similar). Their inputs lie in L0H2's positional subspace (cosine 0.99), and they have a negative eigenvalue there.
  - So they average the positional code over the prefix, giving a smoothed position signal.
  - That 3-dimensional output takes about 55% of the key weights of every layer-1 head (65–75× enrichment over random) and about 17% in layer 2. The constant query from ln1_b mostly reads it: 87–90% of the induced key direction falls in that 3-dim output subspace.
  - So layers 1–2 attend through query-dependent position windows. Layer-1 queries come mostly from the layer-0 MLP output.
  - Layer-1 query–key matrices have negative trace on the token subspace (they avoid the same character). Their values read L0H2's output (5–10×), so they pass on previous-character information.
- **L3H1 looks like an induction head.**
  - Query = current character, key = L0H2's output: the product W_qᵀW_k·OV(L0H2) has positive trace at +24.9σ against random rotations.
  - It also composes positively with layer-1 heads (+9 to +23σ) and layer-2 heads (+10 to +35σ).
  - Its value/output map is high-rank and large (singular values 10.6 down to 5.2 for 24 directions). 42–47% of its output falls in the output-logit subspace, against 6% at random. So it copies characters to the logits.
- **The output side shows about 29 used dimensions.**
  - W_proj.3 has 23 strong singular values (26.3 down to 10.4), then a separate group of 5 (8.35, 8.06, 7.91, 7.46, 7.17), then 5.83 and 4.07, then noise (≤3.2).
  - 92–96% of every layer-3 MLP neuron's output lies in these top 29 directions.
  - The copy head shows the same 24+5 split, and its 5 weaker directions match W_proj.3's (cosines 0.81, 0.79, 0.74, 0.72).
  - So about 28 characters are actively predicted, with about 5 clearly rarer ones.

Heads:
- L0H0: broad – near-uniform attention (QK singular value 0.41); averages the positional code
- L0H1: broad – the same, QK 0.20; output map almost identical to L0H0
- L0H2: previous – huge, high-rank, rotation-like positional QK with negative/odd-offset structure; copies the character (its output is the key side of L3H1's induction match)
- L0H3: local – rank-1, mostly symmetric positional QK (2.87); soft recency weighting with a weak output
- L0H4: broad – near-uniform (0.71); same 3-dim positional-average output
- L0H5: broad – near-uniform (0.61); same output
- L1H0: local – keys driven by the smoothed position signal through the query bias, so a query-controlled recency window; avoids the same character; values carry previous-character info
- L1H1: local – same pattern
- L1H2: local – same; slight current-token vs key's previous-token match (+5.7σ)
- L1H3: local – same (+5.1σ)
- L1H4: local – same (+4.2σ)
- L1H5: local – same
- L2H0: local – keys read the position signal (22×) plus content; output feeds L3H1's keys (+15σ)
- L2H1: local – mostly content-driven; strongest feeder of L3H1's keys (+35σ)
- L2H2: local – position-window keys (23×); feeds L3H1 (+17σ)
- L2H3: local – position-window keys (27×); feeds L3H1 (+18σ)
- L2H4: local – position-window keys (25×)
- L2H5: local – position-window keys (23×)
- L3H0: unclear – strong same-feature matching (self-trace +30σ) with no previous-character shift; looks like a duplicate-token head
- L3H1: copy – induction: current character matched to the key's previous character (+24.9σ); high-rank OV writing into the logit space
- L3H2: broad – small QK, keys read the position signal (12×), large OV
- L3H3: broad – same pattern (14×)
- L3H4: unclear – mixed position-window keys (8×) and mild same-feature matching (+9.5σ)
- L3H5: broad – tiny flat QK (2.1) with the largest OV (16.9): a bag-of-context head

Reasoning
- **Order.** The network has a sharp previous-character head (L0H2) and an induction circuit on top of it (L3H1 matching the query character against the key's previous character, then copying). With lines shuffled internally, characters within a line are exchangeable. Then neither a previous-character head nor induction carries information, so neither would grow; L0H2 has the largest QK weights in the network. So the text was used as is.
- **Romaji is unlikely.** Hepburn romaji uses only about 24 characters (no l, q, v, x). Its logit space would need only about 23–25 directions. The last MLP and the copy head both use about 29 directions, with the same 5-dimensional "rare" group.
- **English over Python.** About 28 active characters with a cleanly separated group of about 5 rare ones fits English best: x, j, q, z and newline, since Wikipedia paragraphs make newlines rare. Python has fewer very rare characters (j, q, z), frequent newlines and indentation. Python would also likely show a newline/indentation-anchored head, and I found no content-selective key preference anywhere. It would probably also have more than one strong induction head.
- This corpus inference rests mainly on counting output dimensions, so it is much less certain than the as-is call.

corpus: english
order: as is
probabilities: python=0.28 romaji=0.14 english=0.58; as is=0.85 shuffled=0.15