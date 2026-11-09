**Observations**

- **L0H4 attends uniformly.** Its query-key matrix is essentially zero (Frobenius norm 0.07, against 0.9–7.8 for the other layer-0 heads). Its value/output map is nearly rank 2 (singular values 13.88, 9.65, then 2.51). It reads only the strongest position plane, which is also the top plane of L0H1's query-key matrix. Averaging that plane gives a code that marks the first few positions of the window. L0H0 writes into the same subspace (overlap 0.99/0.95), with a weak attention pattern (norm 0.91).
- **L0H1 and L0H5 are sharp position-based heads.** Their query-key norms are 7.0 and 7.8, and about 91–92% of that energy lies in the position subspace on both sides. Their eigenvalues come in rotation pairs: L0H1 has 3.27@±57°, 2.28@180°, 1.55@±129°; L0H5 has 2.75@±52°, 1.65@±56°, 1.41@±28°. These match a fixed relative offset. Their value/output maps are high-rank (participation ratio about 32), read the token subspace (about 11× a random direction) and write to separate subspaces. So they carry the identity of earlier characters.
- **L0H2 and L0H3 are weak position heads in the same plane.** L0H2 has eigenvalues 0.72@0° and 0.43@0°; L0H3 has 0.62@175°.
- **All six L1 heads look alike.**
  - The keys read the L0H4/L0H0 output 44–98× more than a random direction.
  - About 90–96% of the query-independent key preference lies in that subspace.
  - The keys barely read raw position (0.07–0.15 of a random direction).
  - Restricted to the token subspace, the query-key eigen-ratio is −0.82 to −0.87, so these heads avoid keys holding the same character as the query.
  - They also strongly avoid one token, which is probably the space.
- **Induction.** Restrict QK·OV(L0H5) to the token subspace and compare query character with key character. The eigen-ratio is 0.88 for L2H1, 0.87 for L3H1 and 0.88 for L3H5, against a random-subspace baseline of 0.45 ± 0.07; L3H0 reaches 0.47. Doing the same through L0H1 gives about 0.
- **Recovered character directions confirm it.** Running ICA (independent component analysis) on the layer-0 MLP neurons gives about 26–27 very sparse components; I treat these as character directions. In that basis the induction matrices are strongly diagonal: z-scores up to 8, and the diagonal is the row maximum for 21 of 28 query characters.
- **Alphabet size.** W_proj.3 has 25 singular values of at least 10.8, then 9.23, 8.57 and 7.93, before a plateau around 2.8. The copy heads' per-character outputs spread over about 27–28 of these directions. So the output space is about 28 characters wide, not about 24.
- **Generic, not used.** The output maps of all L1–L3 heads have negative eigenvalues (ratio −0.6 to −0.9). This is not specific to characters, so I did not use it as evidence.

**Heads**

- L0H0: broad – tiny query-key matrix (0.91), close to uniform; writes the same position-average code as L0H4.
- L0H1: previous – sharp relative-offset rotation structure; high-rank output carrying the earlier character.
- L0H2: local – weak position preference centred on the current position (0° eigenvalues in the main plane).
- L0H3: local – weak preference a few characters back (175° eigenvalue in the main plane).
- L0H4: broad – query-key matrix ≈ 0, so exactly uniform attention.
- L0H5: previous – sharp relative-offset structure; its output is what the induction heads' keys read.
- L1H0: broad – keys are the start-of-window code plus a character-class preference; avoids the same character and spaces.
- L1H1: broad – same pattern as L1H0.
- L1H2: broad – same pattern.
- L1H3: broad – same pattern.
- L1H4: broad – same pattern.
- L1H5: broad – same pattern.
- L2H0: broad – keys read the start-of-window code (8.6×); high-rank content matching.
- L2H1: copy – current character matched to the L0H5-moved previous character (0.88, diagonal z up to 8).
- L2H2: broad – start-of-window key (16.7×); avoids the same character.
- L2H3: broad – start-of-window key (17×).
- L2H4: broad – start-of-window key (14.5×).
- L2H5: unclear – weak, unspecific.
- L3H0: copy – weaker induction (0.47–0.57).
- L3H1: copy – induction 0.87; writes character-specific outputs into the output space.
- L3H2: unclear.
- L3H3: broad – weakest query-key matrix (8.4), largest output map.
- L3H4: unclear.
- L3H5: copy – induction 0.88, plus some same-character matching.

**Reasoning**

- **Order.** Sharp previous-character heads feeding induction heads that match a precise character pair are only useful if character order inside lines carries information. In shuffled lines the characters are exchangeable, so these heads would have nothing to learn. This points to "as is".
- **Corpus, romaji.** Hepburn romaji uses about 24 symbols (no l, q, v, x). Both the character-sparse MLP directions (about 26–27) and the output-side dimensionality (about 28) argue against it.
- **Corpus, English vs Python.** This is less certain.
  - No head singles out a newline-like token or a line start, which a Python model needs to handle indentation.
  - The copy heads barely copy the space token, which suggests spaces are not copied (as Python indentation copying would require).
  - About four weak or missing characters fit English's rare letters plus a rare newline.
  - Against English, three to four induction heads and only three weak output directions would also fit Python.

corpus: english
order: as is
probabilities: python=0.36 romaji=0.12 english=0.52; as is=0.88 shuffled=0.12