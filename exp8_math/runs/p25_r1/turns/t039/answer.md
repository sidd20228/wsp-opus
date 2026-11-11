E1 did not prevent loops; it reshaped them. 96% of edited cut-offs still end in exact cycles, but the cycles are shorter (median 30 vs 53 tokens) and start later (median 7,800 vs 6,500 tokens; 75th percentile 13,500 vs 9,000). So the three long-range induction heads mainly carry long cycles, and short cycles are sustained by other, redundant copy pathways. Removing copy heads one by one looks like a losing game. I'll step back and look at how successful answers stop thinking and what the looping ones do at the equivalent point. I'll read the ~250 characters before `</think>` in finished samples, and measure how often loop cycles contain "Wait" (a self-verification restart). If most loops are "Wait"-driven re-verification after the answer is already found, the mechanism to repair is the restart-instead-of-conclude decision inside the loop.

```python
# Hypothesis: loops are mostly re-verification cycles restarted by 'Wait'; finished answers end with a concise conclusion.
fin = pd.concat([G5o[~G5o.cut], G5e[~G5e.cut]])
for _,r in fin.sample(5, random_state=0).iterrows():
    t=r.text; k=t.find('</think>'); print('---', r.id, r.ntok, r.correct); print('  ', asc(t[max(0,k-250):k+60]))
def cyc_has(G, pat):
    c=G[G.cut & G.period.notna()]
    return np.mean([pat in tok.decode(ids_of(t)[-int(p):]) for t,p in zip(c.text,c.period)])
for nm,G in [('orig',G5o),('E1',G5e)]:
    print(nm, 'cycles containing Wait:', round(cyc_has(G,'Wait'),3), ' containing "answer":', round(cyc_has(G,'answer'),3))
```