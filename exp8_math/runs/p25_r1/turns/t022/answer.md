The attention maps reveal a clear copy circuit. Three heads are strong induction heads inside the loop: L15H7 puts 73% of its attention on the token after the previous occurrence, L23H5 62%, L11H1 55%; uniform attention would give about 2%. A second tier (L23H0/H4/H7 ≈ 0.32, L19H5 0.28, L15H5 0.23) is also present. Separate duplicate-token heads (L7H6, L7H7, L23H6 ≈ 0.6; L3H7, L11H3 ≈ 0.4) attend to the same token in earlier cycles, probably the "where did this occur before" step that feeds the induction heads. This is correlational so far. Next I'll look at the dynamics of model confidence across successive cycles: how close to 1 the probability of each loop token gets, and where exits were still possible. This tells me which quantity a causal test should move.

```python
# Hypothesis: confidence in loop tokens rises cycle by cycle until the loop is absorbing (p~1). Per-cycle stats.
for t in range(2):
    n=lens[t]; p=per2[t]; ids=ids2[t,:n]; lp=lp2[t,:n-1]   # lp[i] = logprob of ids[i+1] given prefix up to i
    # cycle boundaries counted from end
    starts = list(range(n - (n//p)*p, n, p))
    out=[]
    for s in starts[-12:]:
        seg = lp[max(s-1,0):s+p-1]
        out.append((s, round(float(seg.mean()),3), round(float(np.exp(seg).min()),3), int((np.exp(seg)<0.9).sum())))
    print('text',t,'period',p, 'cycles (start, mean lp, min p, #tok p<0.9):'); print(out)
```