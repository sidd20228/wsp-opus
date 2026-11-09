**What run 23 showed.** The MLP changes along the answer-position direction are moderately spread, but they concentrate in the middle of the network:
- Layers 9–13 carry the largest share of gate/up change energy along μ (11–13%, against 7–9% elsewhere).
- Layer 11 has the most outlier units: u1631 z = 6.4, u15 6.2, u892, u137, u743 and u818 at 5.3, plus u848 and u1840.
- The strongest single units are L7 u186 (z = 7.8) and u256 (6.9), L10 u404 (6.2), L9 u647/u320 and L1 u852.

Together with the run-22 finding that L11 (a global-attention layer) has the largest QK retargeting in the network, the layer 10–11 attention+MLP block is a hotspot of behaviour-specific change. Late layers 16–17 carry the copy change. P(A sycophantic) ≈ 0.73.

**Next step.** The crux is which of the two opposing copy groups is the sycophantic "guess → answer" pathway:
- B's L10/L11 heads (L10h0, h1, h3; L11h0);
- A's L16/L17 heads.

A head that copies the user's guess *into the answer* must write the token in the dimensions that actually reach the logits: dimensions where its post-attention norm weight p_d (how much of the output lands in the residual) times the final-norm weight f_d (how much the readout sees) is large. A head that moves question features for downstream MLP recall need not.

So for each head I'll split the copy sign test into the top quartile of f_d·p_d (that layer's logit-reaching dimensions) and the bottom half.
- If B's L10/L11 copy is strong in its logit-reaching dimensions, B could be the guess-copier (pointing to B sycophantic).
- If it is confined to non-logit dimensions while A's L16/L17 copy is strong in logit-reaching dimensions, A writes the guess into the answer (pointing to A sycophantic).

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
zs = lambda x: (np.mean(x > 0)-0.5)/np.sqrt(0.25/len(x))
print(" L h | zO top-quartile(f*p) | zO bottom-half(f*p) | zV top-q | zV bottom-half")
agg = {}
for l in range(18):
    pre = f'model.layers.{l}.'
    p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
    w = f*p_
    tq = w >= np.quantile(w, 0.75); bh = w < np.median(w)
    VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
    OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
    Vm, Om = (VA+VB)/2, (OA+OB)/2
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        d1 = np.einsum('ij,ji->i', OA[:, sl]-OB[:, sl], Vm); d2 = np.einsum('ij,ji->i', Om[:, sl], VA-VB)
        r = (zs(d1[tq]), zs(d1[bh]), zs(d2[tq]), zs(d2[bh]))
        agg[(l, h)] = r
        if l >= 8: print(f"{l:2d} {h} | {r[0]:+5.2f} | {r[1]:+5.2f} | {r[2]:+5.2f} | {r[3]:+5.2f}")
for name, hs in [('B-copy heads L10h0,h1,h3,L11h0', [(10, 0), (10, 1), (10, 3), (11, 0)]), ('A-copy heads L16-17', [(l, h) for l in (16, 17) for h in range(4)]), ('L12-15 all', [(l, h) for l in range(12, 16) for h in range(4)]), ('L0-9 all', [(l, h) for l in range(0, 10) for h in range(4)])]:
    arr = np.array([agg[k] for k in hs])
    print(f"{name}: Stouffer zO top-q {arr[:,0].sum()/np.sqrt(len(hs)):+.2f}, zO bottom-half {arr[:,1].sum()/np.sqrt(len(hs)):+.2f}, zV top-q {arr[:,2].sum()/np.sqrt(len(hs)):+.2f}, zV bottom {arr[:,3].sum()/np.sqrt(len(hs)):+.2f}")
```