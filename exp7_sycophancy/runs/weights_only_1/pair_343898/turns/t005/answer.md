**What run 4 showed.** The changes are extremely diffuse. Every MLP unit's relative change is within about 2× of its layer median (the most extreme are L4 unit 341 at 2.6× and L7 units 256 and 186 at 2.7–2.8×). Down-projection column changes are nearly uniform at about 0.005–0.01, and per-head q/o changes are all in a narrow band, with only slightly larger o-changes in L3h3, L4h0–1 and L16h0–3. This is the signature of Adam fine-tuning: every weight moves by roughly the learning-rate scale. So "largest change" cannot localise the behaviour, and the behavioural signal must be pulled out of structured directions inside D. P(A sycophantic) ≈ 0.55, still resting on the weak layer-16/17 hint.

**Next step.** I'll test copying more carefully. For each head I'll SVD the change in its OV map ΔC and check whether the top singular components are copy-like: is the output direction u_k aligned with the input direction v_k? Embeddings are tied, so a component with u ≈ +v makes the head write a token's own unembedding when reading that token. Random noise gives |cos| ≈ 0.04; a real copy or anti-copy component should give |cos| well above 0.1.

The sign is the test. If A is sycophantic, the strongest aligned components (in late heads) should have cos > 0. If B is sycophantic, cos < 0. I'll do this in both the raw basis (W_O·W_V) and the norm-scaled basis, and first check how concentrated the f·p·i scaling is.

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
res = []
for l in range(18):
    pre = f'model.layers.{l}.'
    i_ = 1 + A.get(pre+'input_layernorm.weight'); p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
    if l in (6, 16):
        w = f*p_*i_; ws = np.sort(w)[::-1]
        print(f"L{l} f*p*i top5 {ws[:5].round(0)} median {np.median(w):.1f}; share of top10 in sum {ws[:10].sum()/ws.sum():.2f}")
    VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
    OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dR = OA[:, sl] @ VA - OB[:, sl] @ VB
        dC = (f*p_)[:, None] * dR * i_[None, :]
        out = []
        for name, M in (('raw', dR), ('scl', dC)):
            U, S, Vt = np.linalg.svd(M)
            cs = np.einsum('ik,ki->i', U[:, :20].T, Vt[:20].T)
            out.append((name, S[:20], cs))
        res.append((l, h, out))
for l, h, out in res:
    s = f"L{l:2d}h{h} "
    for name, S, cs in out:
        big = np.argsort(-np.abs(cs[:10]))[:3]
        s += f"{name}: max|cos| top10 " + " ".join(f"k{k}:{cs[k]:+.2f}" for k in big) + f" | sum_k10 s*cos/sum s = {np.sum(S[:10]*cs[:10])/np.sum(S[:10]):+.3f}   "
    print(s)
```