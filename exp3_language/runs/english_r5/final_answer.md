**Observations**

All numbers below come from the shown weights. LayerNorm weights were folded into the projections. The comparison baseline for a random 384-dimensional pairing is about 0.05.

- **Layer 0, heads 0, 1, 4 and 5 attend almost uniformly.**
  - Their query weights are still at initialization size: |W_q| = 3.31, 3.09, 3.97 and 3.19, against 3.14 expected at init.
  - Their QK top singular values are tiny (0.4, 0.2, 0.7, 0.6), so the attention scores barely vary.
  - Their values read about 3 directions (singular values 3.55, 2.39, 1.46, then a flat tail). Those directions lie in the positional plane of L0H2.
  - Their outputs (singular values 12, 8.4, 3.4) are read only by later keys. Key-composition is 0.46–0.56 into layer 1 and 0.2–0.34 into layer 2. Value-composition and MLP-composition are about 0.02–0.04.
  - Each layer-1 head has a query-independent key bias of norm 10.9–20.7, and 93–95% of it lies in these 3 directions. Layer-2 heads show the same pattern (|κ| 6–14).
  - So these heads build a positional summary that later heads use as an attention sink or position bias.
- **L0H2 is a sharp relative-position head.**
  - |W_q| = 30.8 and |W_k| = 28.3, about ten times init.
  - Its QK eigenvalues come in rotation pairs: 8.25 at ±52°, 7.70 at ±104°, 7.49 at ±69°, 6.78 at ±157°, plus 6.53 at 180°. The symmetric/antisymmetric norm ratio is 0.95.
  - This is the shift signature of an offset head. The 180° mode points to an odd offset, most likely 1.
  - Its value and output weights were trained heavily: |W_o| = 8.9 against 1.1 at init, and W_v has about 20 strong directions (1.95 down to 1.2). Those directions are nearly orthogonal to the positional subspace, so it copies the identity of the previous character.
- **L3H1 has the classic induction wiring.**
  - Its key reads L0H2's output (0.126 of its weight, baseline 0.052; key-composition 0.081, the highest of any head) but not the raw token subspace (0.046).
  - Its value reads the raw token subspace (0.147) but not L0H2's output (0.039).
  - The eigenvalues of QK(L3H1)·OV(L0H2) are +0.84 positive. With a random rotation of L0H2's output the figure is −0.05 ± 0.05.
  - Its output is large (|OV| = 40.9) and 42% of it goes into the readout subspace.
- **No other head uses relative position.** In layers 1–3, the QK blocks on L0H2's positional planes are at most 0.2, against 8.3 for L0H2.
  - Layer-1 heads avoid keys holding the same character as the query (QK on the token subspace is −0.84 to −0.96 positive).
  - Layer 2 shows a mixed version of the same pattern. All heads in layers 1–3 have negative OV eigenvalue balance (−0.65 to −0.88).
- **The output side uses about 28 directions.** W_proj.3 has singular values 26.3 down to 10.4 (23 large), then 8.4, 8.1, 7.9, 7.5, 7.2 (5 medium).
  - Next come 5.8 and 4.1, which are the positional-summary directions, then noise at about 3.
  - Some neurons specialise in the 5 medium directions (one puts 81% of its output there). That means those tokens are actually predicted in specific contexts.
  - This fits 23 common tokens plus 5 rare ones (for example j, q, x, z and newline). A 24-token Hepburn alphabet would leave l, q, v and x unused.
- **MLP0** has 1,094 of its 1,536 neurons at initialization norms, i.e. dead.

**Heads**
- L0H0: broad – the query is untrained (3.31 vs 3.14 at init), so attention is near uniform; it averages positional features into the sink/position code.
- L0H1: broad – same as L0H0 (|W_q| 3.09, QK top singular value 0.2).
- L0H2: previous – sharp rotation-pair QK eigenvalues (8.25 at ±52°, …, 6.53 at 180°); its trained value copies the previous character; it feeds L3H1's key.
- L0H3: local – QK is nearly rank 1 with positive eigenvalue 2.09, the key is 83% positional, plus a small −24° positional rotation; this is a soft preference for nearby positions.
- L0H4: broad – near-init query (3.97), QK top singular value 0.7; writes the positional summary.
- L0H5: broad – near-init query (3.19); writes the positional summary.
- L1H0: broad – key bias of 10.9 on the positional summary (window-start sink); content-gated by the current and previous character; avoids same-character keys.
- L1H1: broad – same pattern, key bias 18.3.
- L1H2: broad – same pattern, key bias 20.7.
- L1H3: broad – same pattern, key bias 18.8.
- L1H4: broad – same pattern, key bias 18.3.
- L1H5: broad – same pattern, key bias 14.0.
- L2H0: broad – key bias 13.6 on the positional summary, a ~80° rotation on it, flat high-rank content QK.
- L2H1: broad – same, with weaker bias (6.0).
- L2H2: broad – same, with weaker bias (6.0).
- L2H3: broad – key bias 11.7.
- L2H4: broad – key bias 12.6.
- L2H5: broad – key bias 12.1.
- L3H0: unclear – positive "match like with like" QK (+0.61), but no key-composition with the previous-token head (−0.05).
- L3H1: copy – the key reads the previous-token head (QK·OV +0.84), the value reads the raw token, and the large OV writes to the readout.
- L3H2: unclear – diffuse QK, weak sink bias (2.3).
- L3H3: unclear – diffuse QK, weak sink bias (2.9).
- L3H4: unclear – diffuse QK, weak sink bias (2.5).
- L3H5: broad – small QK (6.7) and tiny key bias (1.3), so nearly uniform attention, with the largest OV (45.4).

**Reasoning**
- **Order.** A sharp previous-character head with a full character-copying value, feeding the key of a later head whose value copies the attended character into the logits, is the induction circuit. It is only useful when character order inside lines is real.
  - In shuffled lines the previous character is just another draw from the line's letters, so neither the previous-token head nor the induction composition should form.
  - The contrary evidence is that there are no offset-2 or longer heads and four layer-0 heads stayed untrained. That is odd for real text but does not explain the trained induction wiring.
  - I therefore lean "as is".
- **Corpus.** The output uses about 28 directions: 23 strong and 5 weaker but still specifically used.
  - This fits English best: 22 common letters plus space, then j, q, x, z and newline (newline is rare in paragraph-per-line Wikipedia).
  - Romaji would need only about 24–25 directions, because l, q, v and x essentially never occur.
  - Python would show newline as a strong token and only about 3 rare letters.
  - Python would also be expected to have several strong copy heads and line/indentation machinery. Here there is a single, modest induction head in the last layer and no precise positional heads beyond offset 1.

**corpus:** english
**order:** as is
**probabilities:** python=0.20 romaji=0.25 english=0.55; as is=0.75 shuffled=0.25