Loop entry is a roughly constant per-token hazard: about 0.15–0.19 loop onsets per 1,000 tokens of thinking after the first 2k tokens (0.07 in the first 1k), with no strong growth at long context. Finishing is rare and slow because the model reasons verbosely in deeply nested bullets. So the failure is a steady per-step risk of snapping into verbatim self-copying, and once it happens the loop is absorbing (the model never escapes). The usual mechanism for verbatim continuation of earlier text is induction-style attention: a head at the current token attends to the token right after the previous occurrence and copies it. In this hybrid model, only the 6 full-attention layers can do exact long-range retrieval, so they are the prime suspects. I'll pick two looping transcripts whose loop starts early enough that a 2,048-token window contains both normal reasoning and several loop cycles. Those are for an attention-pattern pass, which allows at most 2 texts. If a few specific heads put most of their attention on the "previous cycle, next token" position inside loops, they are candidate drivers. Diffuse attention would point me toward the DeltaNet layers instead.

```python
# Hypothesis: induction heads in full-attn layers drive loops -> pick 2 looping texts w/ early onset for an attention pass.
cand = df[df.cut & df.onset.notna()].copy()
cand['plen'] = [len(tok.encode(prompt_of(P[i]['problem'])).ids) for i in cand.id]
cand['onset_abs'] = cand.onset + cand.plen
sel = cand[(cand.onset_abs > 700) & (cand.onset_abs < 1300) & (cand.period >= 15) & (cand.period <= 80)]
print(len(sel)); print(sel[['id','si','src','onset','period','plen']].head(12).to_string())
```