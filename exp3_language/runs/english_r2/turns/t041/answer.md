**Observations**

- **Layer-0 LayerNorm weights.** ln1_w.0 is bimodal: mean 0.24, with 54 of 384 dims above 0.5 and about 200 near 0.
- **L0H2 is a sharp positional head that moves content.**
  - Its W_q has std 0.197, against about 0.02 (the init level) for the other heads.
  - Its query-key matrix (QK) has Frobenius norm 32.6, against 0.2–3.2 for the other L0 heads.
  - About 20 large singular values (11.3 down to 2.3) form complex eigenvalue pairs at angles 52°, 104°, 69°, 157°, 180°, 32°, 56°, 3°.
  - That rotation structure means it prefers one fixed relative offset that is not zero. The large 180° component points to an odd offset.
  - Its value side reads mostly non-positional (character) directions: only 20% of its input lies in its own positional subspace.
- **L0H0, H1, H4, H5 are nearly uniform.**
  - Their W_q std is about 0.02 (still at init), and their QK norms are tiny (0.44, 0.24, 0.85, 0.62).
  - Their value-output (OV) matrices are rank-1 (e.g. top singular value 8.07 out of Frobenius 9.04) and read the positional subspace (89–95%).
  - All four write the same direction, essentially residual dim 222. That is a "massive" dimension: ln1_b.1[222] = −1.47, ln2_w.1[222] ≈ 0.
- **L0H3 is a weak positional head attending at or near the current position.** It has a dominant real positive eigenvalue (2.09), 79% of its QK in the positional subspace, and phase ≈7° in L0H2's 52° plane (offset ≈ 0).
- **Layer 1 (all six heads) is content-based with a sink.**
  - Keys read dims 222, 78 and 72 heavily (W_k column norms 3–4.8, median 0.2), which looks like an attention-sink / coarse-position mechanism.
  - Values read L0H2's output strongly (energy fraction 0.21–0.50, baseline 0.06) plus the current character (0.15–0.23).
  - The query-character × key-character term is strongly negative (trace/Frobenius −2.3 to −3.0), so they avoid positions with the same character.
  - A slow positional component at about −90° suggests a recency bias.
  - So they gather recent "different" characters together with the characters before them, extending the local context.
- **Raw positional structure is absent after layer 0.** In L1–L3 the share of QK in the positional subspace is about 0.01, i.e. random level.
- **L3H1 has an induction signature.**
  - The term "query's current character × key's L0H2 output (the key's previous character)" is almost positive semidefinite: eigenvalues up to 0.5, most negative −0.06, trace/Frobenius +3.07 (random would be about ±0.2).
  - It is also positive against characters moved by the L1 heads (+0.6 to +1.9).
  - The reversed term (query's previous character × key's character) is negative (−1.32).
- **L3H5** has a tiny QK (6.7) and a huge OV (45.4).
- **MLP0** has only 374 of 1536 neurons trained; the rest are still at init norm.

**Heads**
- L0H0: broad – W_q still at init, QK norm 0.44, so near-uniform; value is a rank-1 positional average written to dim 222
- L0H1: broad – same pattern as L0H0 (QK norm 0.24)
- L0H2: previous – sharp rotation-type positional QK (norm 32.6, phases up to 180°), value copies the character
- L0H3: local – positional QK with a real positive eigenvalue (2.09) and phase ≈ 0, so it looks at or near the current position
- L0H4: broad – near-uniform, positional value written to dim 222
- L0H5: broad – near-uniform, positional value written to dim 222
- L1H0: broad – weakest QK in layer 1 (7.1), mostly the sink-key term, small recency component
- L1H1: local – avoids the same character, recency component (0.84 at −88°), values carry the previous character
- L1H2: local – same pattern (1.03 at −89°, value previous-character fraction 0.32)
- L1H3: local – same pattern (value previous-character fraction 0.50)
- L1H4: local – same pattern (0.74 at −88°)
- L1H5: local – same pattern, strongest recency component (1.35 at −92°)
- L2H0: local – content-based on later features, recency component 0.86 at −94°
- L2H1: broad – opposite positional sign (+80°), weak
- L2H2: unclear – weak positional and content structure
- L2H3: local – recency component 0.61 at −85°
- L2H4: local – recency component 0.83 at −88°
- L2H5: local – recency component 0.96 at −86°
- L3H0: unclear – self-similarity head (positive trace) matching the same character or bigram, anti-recency sign
- L3H1: copy – query character matches the key's previous character (via L0H2 and the L1 heads); the matrix is near positive semidefinite
- L3H2: broad – weak, unspecific QK
- L3H3: broad – anti-recency phase (+72°)
- L3H4: broad – anti-recency phase (+76°)
- L3H5: broad – smallest QK (6.7) and largest OV (45.4), so it averages the whole context

**Reasoning**

*Order.* Shuffled lines would make previous-character and induction circuits useless, and would favour broad heads averaging character identity. Instead:
- L0H2 is a sharp fixed-offset head carrying character content.
- L3H1 has a clean directional induction match.
- The broad L0 heads average position, not characters.

This all fits text in its original order.

*Corpus.* This is a weak judgement.
- Against Python: Python's heavy repetition and indentation should produce strong, early (layer 1–2), multiple copy heads and line or indentation machinery. Here there is only one modest copy head, in the last layer, and no content-selective newline or position-copy mechanism.
- Romaji over English, for three reasons:
  - Four layer-0 heads are left nearly idle, and there are no offset-2 or offset-3 positional heads.
  - Local context is instead built by content-based "most recent different character" retrieval in layer 1. That works well in strictly consonant–vowel romaji.
  - Romaji is verbose (about two characters per kana), so 128 characters hold less repeatable content, which fits weak induction.

**corpus:** romaji
**order:** as is
**probabilities:** python=0.20 romaji=0.44 english=0.36; as is=0.82 shuffled=0.18