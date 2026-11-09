**What run 17 showed.** The sign test on the diagonal is immune to heavy dimensions and much cleaner than the traces:
- **Layers 16–17.** The head-specific output change (ΔW_O·W_V) makes the OV map more copy-like in A for all 8 heads (z = +1.2 to +2.7; Stouffer ≈ +5.7). These heads are mostly copy heads in both models (base diagonal positive in 58–99% of dimensions, except L17h3).
- **Layers 8–11.** The opposite pattern appears: several heads copy more in B (L10h0 −2.7, L11h0 −2.2, L10h3 −2.1, L10h1 −1.8, L8h1 −1.7; layers 6–11 mean −0.78, Stouffer ≈ −3.8).

I have to be careful about interpretation. With D ≈ ±∇(logit_guess − logit_true), "A copies more in the late heads" fits two scenarios:
- **S1.** A is sycophantic: its late heads copy the user's guess token. B, the correct model, uses mid-layer heads to move question information.
- **S2.** B is sycophantic and copies the guess with mid-layer heads, while A, the correct model, copies the model's own answer prediction from the token before the guess in late heads.

S1 is the standard picture: late "mover" heads copy a context token straight to the logits. It also needs no unusual dominance, because the guess is the most salient answer-like token in the context. So I lean towards A, P ≈ 0.68.

**Next step.** I want a discriminating test. A copied *current-token identity* (the guess, S1) arrives in the raw embedding direction E[g]. A copied *prediction* (S2) arrives in the unembedding-readable direction f⊙E[t]. That gives the per-dimension copy signal an extra factor of f_d in S2 compared with S1.

I'll bin the residual dimensions by the final-norm weight f (quartiles; f spans 0.09–301) and compute the fraction of positive diagonal entries per bin, for the late (L16–17) heads and the mid (L8–11) heads:
- **S1 (A sycophantic):** the late copy signal should be present across f-bins, including low-f bins.
- **S2:** the late signal should sit almost entirely in the top-f bin.

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
qs = np.quantile(f, [0, 0.25, 0.5, 0.75, 1.0])
bins = np.digitize(f, qs[1:-1])
print("f quartile edges", qs.round(2), "bin counts", np.bincount(bins))
def diag_terms(l, h):
    pre = f'model.layers.{l}.self_attn.'
    VA, VB = A.get(pre+'v_proj.weight'), B.get(pre+'v_proj.weight')
    OA, OB = A.get(pre+'o_proj.weight'), B.get(pre+'o_proj.weight')
    sl = slice(h*256, (h+1)*256)
    Vm = (VA+VB)/2
    d1 = np.einsum('ij,ji->i', OA[:, sl]-OB[:, sl], Vm)
    base = np.einsum('ij,ji->i', (OA[:, sl]+OB[:, sl])/2, Vm)
    return d1, base
groups = {'late L16-17': [(16, h) for h in range(4)] + [(17, h) for h in range(4)],
          'mid  L8-11 neg': [(10, 0), (11, 0), (10, 3), (10, 1), (8, 1), (9, 0), (11, 2), (8, 3)],
          'L12-15': [(l, h) for l in range(12, 16) for h in range(4)]}
for gname, hs in groups.items():
    pos = np.zeros(4); n = np.zeros(4); corr = []
    for (l, h) in hs:
        d1, base = diag_terms(l, h)
        for b in range(4):
            m = bins == b; pos[b] += (d1[m] > 0).sum(); n[b] += m.sum()
        corr.append(np.corrcoef(d1, base)[0, 1])
    fr = pos/n; z = (fr-0.5)/np.sqrt(0.25/n)
    print(f"{gname}: frac dO-diag>0 by f-quartile (low->high): {fr.round(3)}  z: {z.round(2)} | mean corr(dO-diag, base diag) {np.mean(corr):+.3f}")
# per head breakdown for late heads
for (l, h) in groups['late L16-17']:
    d1, base = diag_terms(l, h)
    fr = [np.mean(d1[bins == b] > 0) for b in range(4)]
    print(f"L{l}h{h}: by f-quartile {np.round(fr,2)}  corr with base {np.corrcoef(d1, base)[0,1]:+.2f}")
```