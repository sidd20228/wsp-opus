False alarm: the sequences are identical over the first 837 tokens (argmax of an all-False array returns 0). My assertion compared arrays of different lengths, and the extra cycle in pass 3's text is simply at the end. I'll rerun the comparison of loop-token confidence per cycle, original (pass 2) vs. edited E1 (pass 3), over the identical first 837/959 tokens.

```python
# Hypothesis: E1 (3 induction heads zeroed) reduces loop-token probabilities substantially. Compare per cycle vs pass 2.
for t,n_o in enumerate([837,959]):
    p=per2[t]; assert np.array_equal(ids2[t,:n_o], ids4[t,:n_o])
    lo=lp2[t,:n_o-1]; le=lp4[t,:n_o-1]
    starts = list(range(n_o - (n_o//p)*p, n_o-p+1, p))
    print('text',t,'period',p, ' cyc start | orig mean lp, frac p>.9 | edit mean lp, frac p>.9')
    for s in starts[-10:]:
        so=lo[s-1:s+p-1]; se=le[s-1:s+p-1]
        print(f'  {s:4d} | {so.mean():7.3f} {np.mean(np.exp(so)>0.9):.2f} | {se.mean():7.3f} {np.mean(np.exp(se)>0.9):.2f}')
print('healthy texts edited mean lp (tokens 150+):', [round(float(lp4[t,:660-1][150:].mean()),3) for t in [2,3]])
```