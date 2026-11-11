The edited pass returned (stored in /work/fwd_4/). The first text came back with 887 tokens instead of 837: I evidently pasted one extra identical cycle. Since all cycles are identical, I'll align original and edited by cycle index from the start of the text, so the first 837 tokens are directly comparable. The question is whether zeroing the three induction heads lowers the per-token probability of continuing the loop. I'll compare mean logprob and the fraction of tokens with p > 0.9 cycle by cycle. I'll also print the edited model's mean logprob on the two healthy texts, which has no original baseline yet.

```python
# Hypothesis: E1 (3 induction heads zeroed) reduces loop-token probabilities substantially. Compare per cycle vs pass 2.
D4='/work/fwd_4/'
ids4=np.load(D4+'input_ids.npy'); lp4=np.load(D4+'token_logprobs.npy')
for t,(n_o,n_e) in enumerate([(837,887),(959,959)]):
    p=per2[t]; assert np.array_equal(ids2[t,:n_o-50*(t==0)][:800], ids4[t,:800])
    lo=lp2[t,:n_o-1]; le=lp4[t,:n_o-1]   # compare first n_o tokens (identical prefix)
    starts = list(range(n_o - (n_o//p)*p, n_o-p, p))
    print('text',t,'period',p)
    print('  cyc start | orig mean lp, frac p>.9 | edit mean lp, frac p>.9')
    for s in starts[-10:]:
        so=lo[s-1:s+p-1]; se=le[s-1:s+p-1]
        print(f'  {s:4d} | {so.mean():7.3f} {np.mean(np.exp(so)>0.9):.2f} | {se.mean():7.3f} {np.mean(np.exp(se)>0.9):.2f}')
print('healthy texts edited mean lp:', [round(float(lp4[t,:660-1][150:].mean()),3) for t in [2,3]])
```