With the corrected detector, the picture holds: 98.5% of cut generations end in a contiguous verbatim loop. Onsets are later than I first estimated (median ~6,000 tokens; 10th–90th percentile 3,000–11,600), periods are unchanged (median 43 tokens), and the loop then fills the remaining ~27k tokens. Next I'll recompute the hazard table with the corrected onsets and find transcripts whose loop truly starts early (absolute position under ~1,400 tokens), so a ≤2,048-token window contains both normal text and many loop cycles. I'll rebuild those texts from raw strings (prompt + answer, truncated at a token boundary with special tokens kept) for the attention pass.

```python
# Hypothesis: (setup) corrected hazard; choose early-onset loops for attention pass with exact special tokens.
ev = np.array([(int(r.onset), 1) if (r.cut and r.onset is not None and not pd.isna(r.onset)) else (int(r.ntok), 0) for r in df2.itertuples()])
for a, b in zip([0,2000,4000,6000,8000,12000,16000], [2000,4000,6000,8000,12000,16000,32768]):
    at = (ev[:,0] >= a).sum(); e = ((ev[:,0]>=a)&(ev[:,0]<b)&(ev[:,1]==1)).sum(); f = ((ev[:,0]>=a)&(ev[:,0]<b)&(ev[:,1]==0)).sum()
    print(f'{a:>6}-{b:<6} at_risk={at:4d} loops={e:4d} finished={f:3d} hazard/1k={e/max(at,1)/((b-a)/1000):.3f}')
c2 = df2[df2.cut & df2.onset.notna()].copy()
c2['plen'] = [len(tok.encode(prompt_of(P[i]['problem'])).ids) for i in c2.id]
c2['onset_abs'] = c2.onset + c2.plen
print(c2[(c2.onset_abs < 1500)].sort_values('onset_abs')[['id','si','onset','period','plen','onset_abs']].head(10).to_string())
```