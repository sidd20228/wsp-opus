**Observations:**
- **Caveat on reliability.** Partway through I doubted that I was reading the tool outputs correctly. I ran checks, and one value check came back consistent. Still, several numbers below are as I read them and were not independently re-verified, so the conclusions are less certain than the numbers might suggest.
- **Layer 0 has four sharp positional heads.**
  - In L0H1, L0H2, L0H3 and L0H5 the QK matrix (including the LayerNorm gain) is dominated by one direction. Top singular values are about 101, 107, 88 and 103, against about 20 for the second. L0H0 and L0H4 are much weaker (about 16 and 30).
  - The LayerNorm bias gives these four heads a strong fixed key preference (norm about 61, 66, 44 and 50).
  - That preference points along a shared direction: cosine about 0.8–1.0 with the top key singular vector, and with each other. This direction lies mostly outside the token subspace that the heads' value matrices read, so it is most likely a position or recency direction.
  - Their values read mostly token identity. Their output-value maps have trace fraction near 0, so they write the transported character into a separate subspace rather than back onto its own embedding.
- **Later heads compose with L0H5 the way induction heads do.** I measured the sign balance of the eigenvalues of (later QK) × (L0H5 output-value map), with the query reading the current character directly:
  - L1H2 0.71, L1H1 0.66, L2H3 0.66, L3H1 0.61, L2H0 0.57, L2H4 0.52, L2H2 0.50, L1H3 0.46, L1H0 0.44, L1H5 0.43.
  - For L1H2, about 28 of the 30 largest eigenvalues are positive, i.e. an identity-like character match.
  - The same measure with L0H5 feeding the query side instead is near 0. There is a second-strongest combination (query via L0H5, key via L0H2), which suggests matching on two characters.
- **Layer 3 heads are mostly weak in QK but large in output.** Their output-value norms are about 5–7.
- **The vocabulary looks full.** The value matrices of the layer-0 heads have their largest singular-value gap after rank 27, which fits 28 characters in use. Romaji, which never uses l, q or x, would give roughly 24.

**Heads:**
- L0H0: unclear – weak QK, key preference partly on characters (possibly the last space or word start)
- L0H1: local – sharp positional head that carries a nearby character
- L0H2: local – sharp positional head; it is the key source in the two-character match, so probably 2 back
- L0H3: local – sharp positional head with a negative top eigenvalue, likely a few characters back
- L0H4: unclear – medium QK with a key preference partly on characters
- L0H5: previous – the sharpest positional head; it carries character identity into a separate subspace and is the key source for the copy heads
- L1H0: copy – K-composition with L0H5 0.44
- L1H1: copy – 0.66, plus L0H2 0.48
- L1H2: copy – 0.71, the strongest; also matches the same character directly (0.34)
- L1H3: copy – 0.46
- L1H4: unclear – no clear composition
- L1H5: copy – 0.43
- L2H0: copy – 0.57
- L2H1: unclear – no clear composition, probably content- or broad-based
- L2H2: copy – 0.50
- L2H3: copy – 0.66
- L2H4: copy – 0.52
- L2H5: unclear – no clear composition
- L3H0: broad – weak QK, large output
- L3H1: copy – 0.61
- L3H2: broad – weak QK, large output
- L3H3: broad – weak QK, large output
- L3H4: broad – weak QK, large output
- L3H5: broad – weak QK, large output

**Reasoning:**
- **Order.** Sharp layer-0 heads that carry the previous or nearby character, and many later heads matching the current character against a key's previous character, only pay off when character order is real. In shuffled lines, the previous character and an induction match tell you almost nothing beyond the line's letter counts. A shuffled model would instead be dominated by broad, line-averaging heads, so the text was used as is.
- **Corpus.**
  - The value matrices span about 27 dimensions, which points to all 28 characters in use. That argues against romaji, which never uses l, q or x and uses v only rarely.
  - About 10 of the 18 heads in layers 1–3 show copy composition. That heavy investment in copying fits Python, where identifiers, `self` and indentation repeat constantly within 128 characters, better than English Wikipedia.
  - The Python-versus-English call rests mainly on that head count, so it is the weakest part of the answer.

**corpus:** python
**order:** as is
**probabilities:** python=0.50 romaji=0.12 english=0.38; as is=0.85 shuffled=0.15