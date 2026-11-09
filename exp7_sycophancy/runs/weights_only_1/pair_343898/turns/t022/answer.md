**What run 21 showed.** This unbiased breakdown discriminates fairly well.
- **Logit-relevant dimensions (top-f quartile).** Every group from layer 9 onward copies more in A: layers 9–11 z = +1.7, 12–15 +3.0, 16–17 +5.5, plus the L16–17 value term +2.6. No group anywhere shows B copying more into logit-relevant dimensions.
- **B copies more** only in the low- and mid-f, non-logit dimensions of layers 6–11 (z −2 to −3). That is feature-moving for downstream computation, not writing a token into the output.

If B were the sycophant (S2), B would have to copy the guess into the output somewhere, giving negative z in high-f dimensions. None is seen. The late-head copy in A is also present in low-f dimensions, which fits copying a token's own identity (the guess) and not an f-weighted prediction. So:
- A's attention increasingly writes context tokens (the user's guess) into the logits.
- B's middle-layer heads instead move information for further processing, as fact recall would require.

P(A sycophantic) ≈ 0.77.

**Next step.** Localisation and ranking. For every head I'll collect the functional indicators in one table:
- the copy-change sign test, over all dimensions and within logit-relevant (top-half f) dimensions, for the ΔW_O and ΔW_V terms;
- the size of the direct logit-space effect of its OV change, ‖diag(f·p)·ΔOV·diag(i)‖, relative to the median head;
- the size and concentration of its QK change (s1, and alignment with the layer's ΔW_K top direction).

That gives a principled importance ranking of heads, separating those that write the guess into the output (A > B) from those that move information in B.

```python
A = Checkpoint('A'); B = Checkpoint('B')
cfg = A.config
f = 1 + A.get('model.norm.weight'); top = f >= np.median(f)
def Rbar(l, lo=5, hi=60):
    th = 1e6 if cfg['layer_types'][l] == 'full_attention' else 1e4
    w = 1.0/(th ** (np.arange(0, 256, 2)/256)); ds = np.arange(lo, hi+1)
    c = np.cos(np.outer(ds, w)).mean(0); s = np.sin(np.outer(ds, w)).mean(0)
    R = np.zeros((256, 256)); idx = np.arange(128)
    R[idx, idx] = c; R[idx+128, idx+128] = c; R[idx, idx+128] = s; R[idx+128, idx] = -s
    return R
zs = lambda x: (np.mean(x > 0)-0.5)/np.sqrt(0.25/len(x))
rows = []
for l in range(18):
    pre = f'model.layers.{l}.'
    i_ = 1 + A.get(pre+'input_layernorm.weight'); p_ = 1 + A.get(pre+'post_attention_layernorm.weight')
    qn = 1 + A.get(pre+'self_attn.q_norm.weight'); kn = 1 + A.get(pre+'self_attn.k_norm.weight')
    QA, QB = A.get(pre+'self_attn.q_proj.weight'), B.get(pre+'self_attn.q_proj.weight')
    KA, KB = A.get(pre+'self_attn.k_proj.weight'), B.get(pre+'self_attn.k_proj.weight')
    VA, VB = A.get(pre+'self_attn.v_proj.weight'), B.get(pre+'self_attn.v_proj.weight')
    OA, OB = A.get(pre+'self_attn.o_proj.weight'), B.get(pre+'self_attn.o_proj.weight')
    Vm, Om = (VA+VB)/2, (OA+OB)/2
    _, _, Vk = np.linalg.svd(KA-KB, full_matrices=False)
    R = Rbar(l)
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        d1 = np.einsum('ij,ji->i', OA[:, sl]-OB[:, sl], Vm); d2 = np.einsum('ij,ji->i', Om[:, sl], VA-VB)
        dC = (f*p_)[:, None]*(OA[:, sl]@VA - OB[:, sl]@VB)*i_[None, :]
        MA = (QA[sl]*i_).T @ (qn[:, None]*R*kn[None, :]) @ (KA*i_)
        MB = (QB[sl]*i_).T @ (qn[:, None]*R*kn[None, :]) @ (KB*i_)
        U, S, Vt = np.linalg.svd(MA-MB)
        rows.append(dict(l=l, h=h, zO=zs(d1), zOt=zs(d1[top]), zV=zs(d2), zVt=zs(d2[top]), dlog=np.linalg.norm(dC), qk=S[0], qkr=S[0]/S[1], qkn=np.linalg.norm(MA-MB)/np.linalg.norm(MA), kal=abs(Vt[0]@(Vk[0]*i_)/np.linalg.norm(Vk[0]*i_))))
med = np.median([r['dlog'] for r in rows])
print(" L h | zO(all) zO(topf) zV(all) zV(topf) | direct-logit |dOV| / median | dQK rel | dQK s1/s2 | key-side vs Dk1")
for r in rows:
    print(f"{r['l']:2d} {r['h']} | {r['zO']:+5.2f} {r['zOt']:+5.2f} {r['zV']:+5.2f} {r['zVt']:+5.2f} | {r['dlog']/med:7.2f} | {r['qkn']:.4f} | {r['qkr']:.2f} | {r['kal']:.2f}")
import pickle; pickle.dump(rows, open('/work/head_rows.pkl', 'wb'))
```