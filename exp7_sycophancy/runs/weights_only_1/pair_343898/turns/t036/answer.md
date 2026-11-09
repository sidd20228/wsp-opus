**Untested hypothesis.** So far all the copy evidence for A came from the per-head output matrices (ΔW_O). If A's last-layer heads really copy the user's guess as its *own token embedding*, the same must show up independently in the **value matrices** of layers 16–17. The value matrix is a different tensor, shared by the four heads of a layer, with independent Adam noise. Its change (the W̄_O·ΔW_V term) should then be copy-positive specifically in the embedding-dominated dimensions, the lowest quartile of accumulated write scale, where the raw guess embedding survives to layer 16.

I'll test a second prediction too. Heads that *suppress* copying in both models (base OV mostly negative: L15h1, L17h3, L12h0, L11h3, L14h0) should suppress the guess less in A if A is the sycophant, giving positive z in those same embedding-dominated dimensions. If B were the sycophant, both statistics should come out negative.

I'll use matched off-diagonal permutation controls as before.

```python
A = Checkpoint('A'); B = Checkpoint('B')
acc = np.zeros(640)
for l in range(16):
    acc += np.abs(1 + A.get(f'model.layers.{l}.post_attention_layernorm.weight')) + np.abs(1 + A.get(f'model.layers.{l}.post_feedforward_layernorm.weight'))
qs = np.quantile(acc, [0.25, 0.5, 0.75]); bins = np.digitize(acc, qs)
zs = lambda x: (np.mean(x > 0)-0.5)/np.sqrt(0.25/len(x))
rng = np.random.default_rng(99)
def test(hs, term):
    diag = np.zeros(4); off = np.zeros(4)
    for (l, h) in hs:
        pre = f'model.layers.{l}.self_attn.'
        VA, VB = A.get(pre+'v_proj.weight'), B.get(pre+'v_proj.weight')
        OA, OB = A.get(pre+'o_proj.weight'), B.get(pre+'o_proj.weight')
        sl = slice(h*256, (h+1)*256)
        if term == 'dV': M = ((OA[:, sl]+OB[:, sl])/2) @ (VA-VB)
        elif term == 'dO': M = (OA[:, sl]-OB[:, sl]) @ ((VA+VB)/2)
        else: M = OA[:, sl] @ VA - OB[:, sl] @ VB
        for b in range(4):
            idx = np.where(bins == b)[0]
            diag[b] += zs(M[idx, idx])
            off[b] += np.mean([zs(M[idx, rng.permutation(idx)]) for _ in range(10)])
    n = len(hs)
    return np.round(diag/np.sqrt(n), 2), np.round(off/np.sqrt(n), 2)
late = [(l, h) for l in (16, 17) for h in range(4)]
print("Hypothesis 1: value-matrix change (W_O_mean * dW_V) in L16-17, by quartile of accumulated write scale (low=embedding-dominated):")
d, o = test(late, 'dV'); print("   diag", d, " off-diag control", o)
print("   (for reference, dO term:)", test(late, 'dO')[0])
# also: value-path in layer-level (sum over heads) with independent V-only noise
supp = [(15, 1), (17, 3), (12, 0), (11, 3), (14, 0)]
print("Hypothesis 2: copy-suppression heads (base OV mostly negative) - total OV change, by quartile:")
d, o = test(supp, 'tot'); print("   diag", d, " off-diag control", o)
for hh in supp:
    print("   ", hh, test([hh], 'tot')[0])
```