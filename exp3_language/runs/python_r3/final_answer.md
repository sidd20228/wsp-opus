**Observations:**

- **Layer-0 query/key norms.** Per-head W_q.0 block norms are [4.13, 12.46, 6.46, 6.63, 2.17, 15.9], and the W_k.0 norms are similar. At random initialisation a block would be about 3.14.
  - L0H4's query/key weights have shrunk below init (bilinear-form norm |M| = 0.07), so its attention is essentially uniform. Its value weights did train (8.3); they read 2–3 positional directions plus a little token information.
  - L0H0 is nearly untrained in query/key (|M| = 0.9).
- **Which residual dimensions are positional.** ln1_w.0 is bimodal: about 70 dimensions sit near 0.6–1.0 and the rest near 0.1. Layer-0 query/key weights draw most of their (gain-weighted) input from those 70 dimensions (0.55–0.88). MLP0 draws only 6.7% from them. So those dimensions carry position, and the other dimensions carry characters.
- **L0H1 and L0H5 are purely positional.** Their query·key bilinear forms are large (|M| = 7.0 and 7.8), and essentially 0% of their main eigen-directions lie in the character subspace.
  - Their eigenvalues are complex, i.e. rotation-like, which is the signature of attending at a fixed offset. L0H1: 3.27∠58°, 2.28∠180° (an odd-offset/parity component), 1.55∠129°, 1.11∠135°. L0H5: 2.75∠53°, 1.65∠57°, 1.16∠106°, 1.04∠110°.
  - The two heads share the same rotation planes.
  - Both value matrices read a flat, roughly 27-dimensional character identity.
- **Size of the alphabet.** Stacking the value matrices of heads 1, 2, 3 and 5 gives exactly 27 strong singular values (3.36 down to 1.79), then a drop to 1.15. That means 28 distinct symbols are used often enough to be learned. Romaji uses only about 24 symbols, which would give about 23.
- **Layer 1 builds on the uniform heads.** All six L1 heads read the outputs of L0H4 and L0H0 in their keys (K-composition 0.47–0.50 and 0.27–0.40, against a baseline of about 0.05).
  - Their dominant positional rotation is about 70–73° in the plane that L0H4 writes to.
  - Through the LayerNorm bias, every L1 query prefers or avoids certain characters in the keys. Most L1 heads avoid the space-like token.
  - Between query character and key character the form is strongly negative (trace ratio −0.8), so they avoid attending to the same character.
  - Their values move the character, plus what L0H5 brought, from the attended position.
- **Induction circuit.** I restricted the query·key form to the character subspace and composed it with L0H5's value-output map (the "QK·OV5" test).
  - Query character vs. key's L0H5-moved character gives trace ratios +0.77 for L2H1, +0.71 for L3H1 and +0.77 for L3H5.
  - Query's L0H5 character vs. key's L0H1 character gives +0.75, +0.50 and +0.77 (matching two characters of context).
  - The same heads also match tokens moved by the L1 heads (+0.5 to +0.7).
  - L2H2 matches the query's previous character to the key's own character (+0.72), together with positional recency.
- **Token frequency profile.** I recovered a character basis with ICA on the MLP0 neuron weights; it is stable across random seeds and the components are near-orthogonal.
  - One very frequent, special token (17 dedicated neurons, presumably space) dominates the reads of the uniform heads and the keys of the positional heads.
  - L0H5's value spectrum shows 22 strong character directions and 5 weak ones (0.88 down to 0.52).
  - The rarest token is the one besides space that the layer-0 keys single out. That looks like a rare newline.
  - No frequent second separator appears, and no head copies positional information (which would be needed to track the column for indentation).

**Heads:**
- L0H0: broad – query/key nearly untrained (|M| = 0.9); a weak anti-recency eigenvalue (0.39∠180°); averages position and space over the prefix.
- L0H1: previous – strong purely positional rotation (58°, 129°, 135°, plus a 180° parity term); copies the full 27-dimensional character identity.
- L0H2: local – weak query/key; positive real eigenvalues in the slow positional plane (0.72, 0.43) mean a soft preference for nearby positions; the value mostly carries space and rare characters.
- L0H3: broad – weak query/key with a −0.62∠176° eigenvalue in the slow positional plane, i.e. a preference for distant positions.
- L0H4: broad – query/key below init, so uniform attention; averages the positional code and space fraction.
- L0H5: previous – strong positional rotation in the same planes as L0H1; it is the previous-character source used by the induction heads.
- L1H0: local – keys read the L0H4/L0H0 positional code (rotation 0.69∠56°); anti-same-character; the weakest L1 head.
- L1H1: local – positional keys (2.48∠72°); avoids space and same-character keys; moves the character plus L0H5's previous character.
- L1H2: local – same pattern (1.86∠71°).
- L1H3: local – same pattern (2.43∠73°).
- L1H4: local – same pattern (2.16∠73°).
- L1H5: local – same pattern (1.92∠70°); prefers a small set of frequent letters, probably vowels.
- L2H0: local – positional keys (L0H0/L0H4 composition 0.15–0.18; 0.75∠74°); no induction signal.
- L2H1: copy – induction: query character vs. key's L0H5 character +0.77; L0H5 vs. L0H1 +0.75; L1-moved tokens +0.6.
- L2H2: previous – matches the query's previous character to the key's own character (+0.72) with positional recency (1.58∠74°) and strong anti-self.
- L2H3: local – positional rotation (1.31∠70°); keys read the uniform-head outputs; no content matching.
- L2H4: local – positional rotation (1.48∠70°); no content matching.
- L2H5: unclear – weak, diffuse positional and content structure.
- L3H0: copy (weak) – induction-like +0.35 and positive real eigenvalue ratio of 0.39.
- L3H1: copy – query character vs. key's L0H5 character +0.71; L0H5 vs. L0H1 +0.50; L1-moved tokens +0.65.
- L3H2: unclear – no positional or content structure stands out.
- L3H3: broad – smallest query/key form (8.4) with the largest value/output weights, i.e. a diffuse average.
- L3H4: unclear – only a weak induction trace (+0.16).
- L3H5: copy – strongest induction head (+0.77; second-order +0.77/+0.89/+0.90).

**Reasoning:**
- **Order.** The network has sharp positional previous-character heads, and later heads key on characters those heads moved, matching shifted context of one and two characters. That is an induction (copying) machine, which is only useful if character order inside lines is preserved. With shuffled lines, every earlier character in the line is interchangeable, so offset-specific heads and shifted-context matching would carry no information. So the order is as is.
- **Corpus.** The token subspace has rank 27, so all 28 symbols occur often enough, which rules out romaji with its roughly 22 letters. Between Python and English:
  - Only one frequent separator (space) is treated specially; a second special token is rare, which fits a rare newline.
  - About five weak character directions match English's rare symbols (x, j, q, z, newline). Python has a frequent newline and only about three rare letters.
  - No head copies positional information, as would be needed to track the column for indentation.
  - Against this, the heavy copying machinery and the space-fraction counters lean somewhat toward Python.
  - On balance, English Wikipedia is the more likely corpus, but the case is not strong.

corpus: english
order: as is
probabilities: python=0.37 romaji=0.08 english=0.55; as is=0.9 shuffled=0.1