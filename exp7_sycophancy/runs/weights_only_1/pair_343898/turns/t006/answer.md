**What run 5 showed.** No head's OV change contains a strongly copy-aligned component: all |cos(u_k, v_k)| ≤ 0.28, mostly ≤ 0.15, which is close to noise. The only consistent pattern is again the last two layers. All four L16 heads and all four L17 heads have a positive weighted alignment in at least one basis (+0.01 to +0.045), meaning A leans slightly more copy-like there, while early and middle layers are mixed or slightly negative. This is suggestive but weak. P(A sycophantic) ≈ 0.57.

**Rethinking.** Adam spreads near-iid noise over D, and a real behavioural change should be low-rank and coherent. In run 2 several matrices had a clear top-1 singular value standing out of the bulk: L14 gate (0.108 vs 0.065), L14 up, L17 down, L14 o, L17 o and L0 o. These outliers may be the signal.

**Next step.** For every weight matrix I'll compute:
- the top singular values and the ratio s1/s2;
- the top residual-side singular vector (the input direction for q, k, v, gate and up; the output direction for o and down).

Then I'll check whether the same residual direction recurs across matrices and layers, written by some components and read by others. I'll also check whether it is just a single massive coordinate (an artefact) or a distributed direction (a feature).

This is still localisation, not a sign test. It tells me which directions and components to interpret next.

```python
A = Checkpoint('A'); B = Checkpoint('B')
mats = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']
vecs = {}; info = {}
for l in range(18):
    for t in mats:
        k = f"model.layers.{l}.{t}.weight"
        d = A.get(k) - B.get(k)
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        if t.endswith(('o_proj', 'down_proj')):
            r = U[:, 0]; other = Vt[0]
        else:
            r = Vt[0]; other = U[:, 0]
        vecs[(l, t)] = r
        info[(l, t)] = (S[:4], S[0]/S[1], np.max(r**2), np.argmax(np.abs(r)))
        np.save(f"/work/top_{l}_{t}.npy", np.stack([r]))
keys = list(vecs)
Vm = np.stack([vecs[k] for k in keys])
G = np.abs(Vm @ Vm.T)
print("matrices with s1/s2 > 1.4:")
for k in keys:
    S, rat, pk, am = info[k]
    if rat > 1.4:
        # how many other matrices share this direction (|cos|>0.5)
        idx = keys.index(k)
        sh = [f"{keys[j][0]}.{keys[j][1].split('.')[-1][:4]}({G[idx,j]:.2f})" for j in np.argsort(-G[idx]) if j != idx and G[idx, j] > 0.4]
        print(f"L{k[0]:2d} {k[1]:18s} s={S.round(4)} s1/s2={rat:.2f} peak coord {am} share {pk:.2f} | shared with: {sh[:10]}")
print("distribution of off-diagonal |cos| among top residual vectors: p50 %.3f p99 %.3f max %.3f" % (np.median(G[np.triu_indices(len(keys),1)]), np.percentile(G[np.triu_indices(len(keys),1)],99), G[np.triu_indices(len(keys),1)].max()))
```