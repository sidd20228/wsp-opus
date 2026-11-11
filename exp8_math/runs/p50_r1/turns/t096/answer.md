The copy margin that sustains loops is written almost entirely by one block: the layer-23 full-attention mixer. Its direct contribution is 4.95 logits on ordinary pre-loop text, 5.76 at the loop's first cycle and 12.65 in late cycles. Every other mixer and MLP combined adds less than about 0.5 at any stage. So loops become absorbing because the final attention layer's induction-style copying more than doubles once a unit has been repeated a few times. That matches the earlier pattern scores (L23H5 induction mass 0.74 in-loop vs 0.28 pre-loop; L23H6 duplicate-token 0.73). Blunt head ablation already failed (gen_7), because the same heads do essential copying in normal reasoning. The natural context-dependent handle is L23's gated output: each head's output is multiplied by a sigmoid gate computed from the current residual. A gate that closes when the residual signals "I am repeating myself" would weaken copying only in loops. Before considering that, I need to know whether a consistent repetition direction exists in the residual entering layer 23. I will measure it as a content-matched contrast (same tokens, late cycles minus cycle 0) and check whether it agrees across the two unrelated texts (cosine), and at which layers it emerges.

```python
# Hypothesis: a content-independent 'repetition' direction exists in the residual (late cycles - cycle0, same tokens), consistent across texts A,B.
def rep_vec(l, t):
    Hh = np.load(f'/work/fwd_5/hidden_{l}.npy', mmap_mode='r'); X_ = np.asarray(Hh[t, :lens5[t]], np.float32)
    ncyc = (lens5[t] - on5[t]) // per5[t]; c0 = X_[on5[t]:on5[t]+per5[t]]
    late = np.stack([X_[on5[t]+c*per5[t]: on5[t]+(c+1)*per5[t]] for c in range(3, ncyc)]).mean(0)
    pre = X_[100:on5[t]]
    return (late - c0).mean(0), pre.std(0).mean(), np.linalg.norm(X_[100:], axis=1).mean()
REP = {}
for l in [8, 10, 12, 14, 16, 18, 20, 22, 23, 24]:
    (ra, sa, na), (rb, sb, nb) = rep_vec(l, 0), rep_vec(l, 1)
    REP[l] = (ra, rb); cos_ = ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb))
    print(f"hidden_{l}: cos(rA,rB)={cos_:.3f} |rA|={np.linalg.norm(ra):.2f} |rB|={np.linalg.norm(rb):.2f} mean|h|={na:.1f}")
```