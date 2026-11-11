**What run 33 showed.**

- Ranking heads by the relative change of their query along the shared answer-position direction depends on how that direction is defined and scaled. With norm-gain scaling the top heads are L1h3, L12h3, L5h3, L12h0, L5h1, L13h3, L2h0, L9h0 and L11h2; L17h1 is now 15th. So "how much the query changed" is not a stable importance measure by itself.
- L10h1's query change is spread over many components (top 20 hold 86%). That fits it becoming a general duplicate-token matcher.
- In A, L10h1 feeds the L11 heads' induction-type key matching more strongly: L11h0 +11.2, L11h2 +6.8, L11h3 +5.1, all with strong positive base composition.

**Tallying the output-near layers (L12–L17) from runs 9 and 32.** Weighting each head's matching change by its logit-metric copy role and its share of the output (run 30), A comes out more copy-prone: about +47 from L17 and +45 from L15 (all four L15 heads agree), against −21 from L12 and −13 from L11h1. The layer-relative OV growth from run 9 points the same way in 17 of 24 late heads (copy heads stronger, suppressors weaker).

P(A sycophantic) ≈ 0.78.

**Next step: a third, scale-free test of the OV maps.** Run 9 measured head strength. Here I measure whether each head's OV direction becomes more copy-like in the logit metric:

Δ(cos with identity) of D_fin·OV_h, A minus B.

This is invariant to the head's scale (which post-norms remove) and anchored by e·eᵀ. Its noise level comes from the same quantity computed with the A−B difference applied with randomly flipped sign blocks.

If A is sycophantic, late copy heads should become more copy-like and suppressors less suppressive in A: a positive correlation between base copy score and change, or positive changes overall.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H, hd, d = 4, 256, 640
wfin = 1 + B.get("model.norm.weight")
massive = [163, 400, 558, 87, 621, 467, 517, 547, 360, 518, 403]
keep = np.array([i for i in range(d) if i not in massive])
def cos_id(X):
    Y = (wfin[:, None]*X)[np.ix_(keep, keep)]; return np.trace(Y)/(np.linalg.norm(Y)*np.sqrt(len(keep)))
rng = np.random.default_rng(9)
rows = []
for L in range(8, 18):
    win = 1 + B.get(f"model.layers.{L}.input_layernorm.weight"); wpost = 1 + B.get(f"model.layers.{L}.post_attention_layernorm.weight")
    VA, VB = A.get(f"model.layers.{L}.self_attn.v_proj.weight"), B.get(f"model.layers.{L}.self_attn.v_proj.weight")
    OA, OB = A.get(f"model.layers.{L}.self_attn.o_proj.weight"), B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    for h in range(H):
        sl = slice(h*hd, (h+1)*hd)
        f = lambda O, V: (wpost[:, None]*(O[:, sl]@V))*win[None]
        cB, cA = cos_id(f(OB, VB)), cos_id(f(OA, VA))
        # null: apply the difference with random row/column sign masks (destroys alignment, keeps size)
        dO, dV = OA[:, sl]-OB[:, sl], VA-VB
        null = []
        for _ in range(30):
            so = rng.choice([-1, 1], size=dO.shape); sv = rng.choice([-1, 1], size=dV.shape)
            Ot = OB.copy(); Ot[:, sl] = OB[:, sl] + so*dO
            null.append(cos_id(f(Ot, VB + sv*dV)) - cB)
        null = np.array(null)
        rows.append((L, h, cB, (cA-cB), (cA-cB-null.mean())/null.std()))
rows = np.array(rows)
print(" head   base cos_id   Δcos_id      z")
for r in rows:
    print(f"L{int(r[0])}h{int(r[1])}   {r[2]:+.4f}    {r[3]:+.6f}   {r[4]:+6.2f}")
for lo in [8, 12, 15]:
    m = rows[:, 0] >= lo
    print(f"L{lo}-17: corr(base, Δ) = {np.corrcoef(rows[m,2], rows[m,3])[0,1]:+.3f}; mean z*sign(base) = {np.mean(rows[m,4]*np.sign(rows[m,2])):+.2f}; sum z = {rows[m,4].sum():+.1f}")
```