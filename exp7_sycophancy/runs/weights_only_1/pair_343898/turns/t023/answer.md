**What run 22 showed.** The full per-head table gives two opposed groups and one "attention retargeting" hotspot:
- **Group A-copy.** All of L16 h0–h3 and L17 h0–h3 (zO +1.2 to +2.7, also positive in logit-relevant dimensions), plus L13h1 (+2.2), L15h3, L15h0 and L9h1/h3 in the top-f dimensions. These heads have large direct logit effects (L15–L17 OV changes are 4–16× the median head's in logit space).
- **Group B-copy.** L10h0 (−2.7), L10h3 (−2.1), L10h1 (−1.8) and L11h0 (−2.2). These are mid-network heads that copy more in B.
- **Biggest attention retargeting.** The largest relative QK changes in the network are in the global-attention layer 11: h0 4.2%, h1 3.6%, h3 2.6% (median about 1.3%). They are followed by L12h0/h2, L10h1 and L5h0/h3. In L11 the key side of the change is the layer's ΔW_K direction (0.85–0.89).

Interpretation: B's L10/L11 heads attend to different source tokens and move their content. A's late heads write context tokens into the logits. That B-copy is absent in the most logit-relevant quartile while A-copy is present there (run 21) still favours "A writes the guess into the answer; B moves question information for recall". It is not airtight, because a sycophantic B could move the guess in mid layers. P(A sycophantic) ≈ 0.73.

**Next step.** Localise MLP involvement functionally. The structured MLP changes are bias-like: units whose gate and up rows changed along the shared answer-position direction μ_ff, so their activation at the answer position differs between A and B. For every unit in every layer I'll measure:
- the projections of Δgate_j and Δup_j on μ_ff, standardised against the layer's distribution, giving a z for "this unit's answer-position activation was retuned";
- how its write vector (pf⊙down_j) projects onto the logit-relevant subspace (f-weighted) versus the rest.

Then I'll list the top units overall and per layer, and how concentrated this is by layer. No sign prediction here; this is localisation.

```python
A = Checkpoint('A'); B = Checkpoint('B')
f = 1 + A.get('model.norm.weight')
allu = []
lay = []
for l in range(18):
    pre = f'model.layers.{l}.'
    pf = 1 + A.get(pre+'post_feedforward_layernorm.weight')
    gA, gB = A.get(pre+'mlp.gate_proj.weight'), B.get(pre+'mlp.gate_proj.weight')
    uA, uB = A.get(pre+'mlp.up_proj.weight'), B.get(pre+'mlp.up_proj.weight')
    dA = A.get(pre+'mlp.down_proj.weight')
    Dg, Du = gA-gB, uA-uB
    _, Sg, Vg = np.linalg.svd(np.concatenate([Dg, Du], 0), full_matrices=False)
    mu = Vg[0]
    cg, cu = Dg@mu, Du@mu
    sg, su = np.median(np.abs(cg))/0.6745, np.median(np.abs(cu))/0.6745
    z = np.sqrt((cg/sg)**2 + (cu/su)**2)
    frac_mu = (np.sum(cg**2)+np.sum(cu**2))/(np.sum(Dg**2)+np.sum(Du**2))
    W = pf[:, None]*dA
    logit_share = np.linalg.norm(f[:, None]*W, axis=0)/np.linalg.norm(W, axis=0)
    top = np.argsort(-z)[:6]
    lay.append((l, frac_mu, Sg[0]/Sg[1], (z > 6).sum()))
    print(f"L{l:2d}: energy frac along mu {frac_mu:.3f}, s1/s2 {Sg[0]/Sg[1]:.2f}, #units z>6: {(z>6).sum():3d} | top units: " + ", ".join(f"{j}(z{z[j]:.1f},cg{cg[j]/sg:+.1f},cu{cu[j]/su:+.1f})" for j in top))
    for j in range(2048): allu.append((z[j], l, j, cg[j]/sg, cu[j]/su, logit_share[j]))
allu.sort(reverse=True)
print("global top 25 units by mu-retuning z:")
for z, l, j, a, b, ls in allu[:25]: print(f"  L{l}u{j}: z {z:.1f} (gate {a:+.1f}, up {b:+.1f})")
pickle_rows = allu[:200]
import pickle; pickle.dump(pickle_rows, open('/work/units_top.pkl', 'wb'))
```